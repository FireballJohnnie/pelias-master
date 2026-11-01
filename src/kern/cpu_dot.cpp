// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// Fast CPU row dot products (kern/cpu_dot.h). The dequantizers below compute every value with the same float expression,
// in the same order, as quant/quant_cpu.cpp's dequantize_* (read there, not copied from elsewhere); the summation is ours.
#include "kern/cpu_dot.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

#include "quant/iq_tables.h"
#include "quant/planes.h"
#include "quant/quant_cpu.h"

#if defined(__x86_64__) || defined(_M_X64)
#define PELIAS_KERN_X86 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define PELIAS_TARGET_AVX2
#else
#define PELIAS_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#endif
#endif

namespace pelias::kern {

namespace q = pelias::quant;

namespace {

constexpr int kSeg = 256;    // values per segment (one plane block; one K-quant block; eight 32-value blocks)
constexpr int kLanes = 32;   // summation lanes: value i of a segment goes to lane i mod 32

std::uint16_t u16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }
float h2f(const std::uint8_t* p) { return q::fp16_to_fp32(u16(p)); }

// the 12-byte scale and min field of Q4_K / Q5_K: 6-bit scale and min of 32-value group j (0..7)
void scale_min_k4(int j, const std::uint8_t* s, int* sc, int* m) {
  if (j < 4) {
    *sc = s[j] & 63;
    *m = s[j + 4] & 63;
  } else {
    *sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
    *m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

constexpr std::int8_t kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constexpr std::int8_t kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
std::uint32_t u32le(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) | (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}
constexpr float kIq1Delta = 0.125f;

float e8m0_half(std::uint8_t x) {  // 2^(x - 128): MXFP4's shared scale with its values in half units
  const std::uint32_t bits = x < 2 ? (0x00200000u << x) : (static_cast<std::uint32_t>(x - 1) << 23);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

bool is_fast(q::Type t) {
  switch (t) {
    case q::Type::q8_0: case q::Type::q4_k: case q::Type::q5_k: case q::Type::q6_k: case q::Type::iq4_xs:
    case q::Type::q4_tg: case q::Type::q4_tg2: case q::Type::q4_0: case q::Type::q4_1: case q::Type::q5_0:
    case q::Type::q5_1: case q::Type::iq4_nl: case q::Type::mxfp4: case q::Type::q2_k: case q::Type::q3_k:
    case q::Type::q8_z: case q::Type::f16: case q::Type::bf16: case q::Type::f32: case q::Type::iq2_xxs: case q::Type::iq2_xs:
    case q::Type::iq2_s: case q::Type::iq3_xxs: case q::Type::iq3_s: case q::Type::iq1_s: case q::Type::iq1_m: case q::Type::tq1_0:
    case q::Type::tq2_0: case q::Type::q7_z: case q::Type::q7_zt: case q::Type::i2_s: case q::Type::nvfp4:
      return true;
    default:
      return false;
  }
}

std::atomic<int> g_forced{-1};

bool cpu_has_avx2_fma() {
#if defined(PELIAS_KERN_X86)
#if defined(_MSC_VER) && !defined(__clang__)
  int r[4];
  __cpuid(r, 0);
  if (r[0] < 7) return false;
  __cpuid(r, 1);
  const bool fma = (r[2] >> 12) & 1, osx = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
  const bool f16c = (r[2] >> 29) & 1;
  if (!(fma && osx && avx && f16c)) return false;
  if ((_xgetbv(0) & 6) != 6) return false;
  __cpuidex(r, 7, 0);
  return (r[1] >> 5) & 1;
#else
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#endif
#else
  return false;
#endif
}

// ---------------------------------------------------------------------------------------------------- portable path

// one segment's planes onto w[0..256): plane by plane, s x (+/-1), as quant::plane_add_row adds them
void planes_seg_portable(const std::uint8_t* o, int k_stored, int k_use, float* w) {
  const std::uint8_t* bits = o + 2 * k_stored;
  for (int p = 0; p < k_use; ++p) {
    const float s = h2f(o + 2 * p);
    const std::uint8_t* bp = bits + 32 * p;
    for (int i = 0; i < kSeg; ++i) w[i] += s * (((bp[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f);
  }
}

void fma_seg_portable(const float* w, const float* x, float* acc) {  // acc[32]
  for (int i = 0; i < kSeg; ++i) acc[i % kLanes] = std::fma(w[i], x[i], acc[i % kLanes]);
}

// the fixed tree: lanes 0-7 + 8-15, 16-23 + 24-31, then those two, then 8 lanes as (0+4, 1+5, 2+6, 3+7), (0+2, 1+3), 0+1
float reduce_portable(const float* acc) {
  float a[8], b[8], c[8];
  for (int j = 0; j < 8; ++j) {
    a[j] = acc[j] + acc[8 + j];
    b[j] = acc[16 + j] + acc[24 + j];
  }
  for (int j = 0; j < 8; ++j) c[j] = a[j] + b[j];
  float d[4], e[2];
  for (int j = 0; j < 4; ++j) d[j] = c[j] + c[4 + j];
  for (int j = 0; j < 2; ++j) e[j] = d[j] + d[2 + j];
  return e[0] + e[1];
}

// ------------------------------------------------------------------------------------- the int8-activation path, shared

// One 256-value segment as integers: w_i = sw[i / 16] * q_i - mw[i / 16] (mw 0 for symmetric types).
struct Seg8 {
  alignas(32) std::int8_t q[kSeg];
  float sw[16];
  float mw[16];
};

[[maybe_unused]] bool int_form(q::Type t) {  // every type with an integer form (seg8_portable reads them all)
  switch (t) {
    case q::Type::q8_0: case q::Type::q4_0: case q::Type::q4_1: case q::Type::q5_0: case q::Type::q5_1: case q::Type::iq4_nl:
    case q::Type::mxfp4: case q::Type::iq4_xs: case q::Type::q8_z: case q::Type::q4_k: case q::Type::q5_k: case q::Type::q6_k:
    case q::Type::q2_k: case q::Type::q3_k: case q::Type::tq1_0: case q::Type::tq2_0: case q::Type::q7_z: case q::Type::q7_zt:
    case q::Type::iq2_xxs: case q::Type::iq2_xs: case q::Type::iq2_s: case q::Type::iq3_xxs: case q::Type::iq3_s: case q::Type::iq1_s:
      return true;
    default:
      return false;
  }
}

// the types that take the integer path: an integer form and a vectorized decoder of their integers (the others are faster
// on the float path, which reads the same rounded inputs); the same set on every ISA, so the two paths agree bit for bit
bool int_path(q::Type t) {
  switch (t) {
    case q::Type::q8_0: case q::Type::q8_z: case q::Type::q4_0: case q::Type::q4_1: case q::Type::iq4_nl: case q::Type::mxfp4:
    case q::Type::iq4_xs: case q::Type::q4_k: case q::Type::q5_k: case q::Type::q6_k:
      return true;
    default:
      return false;
  }
}

bool has_mins(q::Type t) {  // the types with a min (offset) per group
  return t == q::Type::q4_1 || t == q::Type::q5_1 || t == q::Type::q4_k || t == q::Type::q5_k || t == q::Type::q2_k || t == q::Type::iq1_s;
}

void set32(Seg8& o, int b, float sw, float mw) {  // one 32-block's scale and min (both of its 16-halves)
  o.sw[2 * b] = o.sw[2 * b + 1] = sw;
  o.mw[2 * b] = o.mw[2 * b + 1] = mw;
}

inline std::int8_t i8(int v) { return static_cast<std::int8_t>(v); }

// the segment's scales and integers, value by value as quant_cpu.cpp / dequant_more.cpp read them
// ints false: the scales only (the AVX2 decoders below fill the integers of their types)
void seg8_portable(q::Type t, const std::uint8_t* p, Seg8& o, bool ints = true) {
  std::int8_t* y = o.q;
  switch (t) {
    case q::Type::q8_0:
      for (int b = 0; b < 8; ++b, p += 34) {
        set32(o, b, h2f(p), 0.0f);
        if (ints)
          for (int j = 0; j < 32; ++j) y[32 * b + j] = static_cast<std::int8_t>(p[2 + j]);
      }
      return;
    case q::Type::q4_0: case q::Type::q4_1: case q::Type::iq4_nl: case q::Type::mxfp4: {
      const int bytes = t == q::Type::q4_1 ? 20 : t == q::Type::mxfp4 ? 17 : 18, qo = t == q::Type::q4_1 ? 4 : t == q::Type::mxfp4 ? 1 : 2;
      for (int b = 0; b < 8; ++b, p += bytes) {
        const float d = t == q::Type::mxfp4 ? e8m0_half(p[0]) : h2f(p);
        set32(o, b, d, t == q::Type::q4_1 ? -h2f(p + 2) : 0.0f);
        for (int j = 0; ints && j < 16; ++j) {
          const int lo = p[qo + j] & 0xF, hi = p[qo + j] >> 4;
          if (t == q::Type::q4_0) { y[32 * b + j] = i8(lo - 8); y[32 * b + j + 16] = i8(hi - 8); }
          else if (t == q::Type::q4_1) { y[32 * b + j] = i8(lo); y[32 * b + j + 16] = i8(hi); }
          else if (t == q::Type::iq4_nl) { y[32 * b + j] = kIq4[lo]; y[32 * b + j + 16] = kIq4[hi]; }
          else { y[32 * b + j] = kFp4[lo]; y[32 * b + j + 16] = kFp4[hi]; }
        }
      }
      return;
    }
    case q::Type::q5_0: case q::Type::q5_1: {
      const bool one = t == q::Type::q5_1;
      const int bytes = one ? 24 : 22, ho = one ? 4 : 2, qo = one ? 8 : 6;
      for (int b = 0; b < 8; ++b, p += bytes) {
        set32(o, b, h2f(p), one ? -h2f(p + 2) : 0.0f);
        const std::uint32_t qh = u32le(p + ho);
        for (int j = 0; j < 16; ++j) {
          const int x0 = (p[qo + j] & 0xF) | static_cast<int>(((qh >> j) & 1u) << 4), x1 = (p[qo + j] >> 4) | static_cast<int>(((qh >> (j + 16)) & 1u) << 4);
          y[32 * b + j] = i8(one ? x0 : x0 - 16);
          y[32 * b + j + 16] = i8(one ? x1 : x1 - 16);
        }
      }
      return;
    }
    case q::Type::iq4_xs: {
      const float d = h2f(p);
      const std::uint16_t sh = u16(p + 2);
      for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((p[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
        set32(o, ib, d * static_cast<float>(ls - 32), 0.0f);
        const std::uint8_t* qs = p + 8 + 16 * ib;
        for (int j = 0; ints && j < 16; ++j) {
          y[32 * ib + j] = kIq4[qs[j] & 0xf];
          y[32 * ib + j + 16] = kIq4[qs[j] >> 4];
        }
      }
      return;
    }
    case q::Type::q8_z: case q::Type::q7_z: case q::Type::q7_zt: {
      const float d = h2f(p);
      std::uint64_t packed = 0;
      for (int b = 0; b < 6; ++b) packed |= static_cast<std::uint64_t>(p[2 + b]) << (8 * b);
      int hslot = 0;
      for (int j = 0; j < 8; ++j) {
        const int lv = static_cast<int>((packed >> (6 * j)) & 63u);
        set32(o, j, t == q::Type::q7_z ? q::q7_z_scale(d, lv) : q::q8_z_scale(d, lv), 0.0f);
        if (t == q::Type::q8_z) {
          for (int k = 0; ints && k < 32; ++k) y[32 * j + k] = static_cast<std::int8_t>(p[8 + 32 * j + k]);
          continue;
        }
        const bool gated = t == q::Type::q7_zt && ((p[8] >> j) & 1u) && hslot < 2;
        std::uint32_t hb = 0;
        if (gated) hb = u32le(p + 233 + 4 * hslot++);
        for (int g = 0; g < 4; ++g) {
          std::uint64_t wv = 0;
          const std::uint8_t* gp = t == q::Type::q7_z ? p + 8 + 7 * (4 * j + g) : p + 9 + 7 * (4 * j + g);
          for (int b = 0; b < 7; ++b) wv |= static_cast<std::uint64_t>(gp[b]) << (8 * b);
          for (int k = 0; k < 8; ++k) {
            const int u = static_cast<int>((wv >> (7 * k)) & 127u), idx = 8 * g + k;
            y[32 * j + idx] = t == q::Type::q7_z ? i8(u - 64) : i8(2 * u - 127 + (gated ? static_cast<int>((hb >> idx) & 1u) : 0));
          }
        }
      }
      return;
    }
    case q::Type::q4_k: case q::Type::q5_k: {
      const bool five = t == q::Type::q5_k;
      const float d = h2f(p), mn = h2f(p + 2);
      const std::uint8_t* sc = p + 4;
      const std::uint8_t* qh = p + 16;
      const std::uint8_t* qs = p + (five ? 48 : 16);
      for (int j = 0; j < 4; ++j, qs += 32) {
        int s1, m1, s2, m2;
        scale_min_k4(2 * j, sc, &s1, &m1);
        scale_min_k4(2 * j + 1, sc, &s2, &m2);
        set32(o, 2 * j, d * static_cast<float>(s1), mn * static_cast<float>(m1));
        set32(o, 2 * j + 1, d * static_cast<float>(s2), mn * static_cast<float>(m2));
        for (int l = 0; ints && l < 32; ++l) {
          y[64 * j + l] = i8((qs[l] & 0xF) + (five && ((qh[l] >> (2 * j)) & 1) ? 16 : 0));
          y[64 * j + 32 + l] = i8((qs[l] >> 4) + (five && ((qh[l] >> (2 * j + 1)) & 1) ? 16 : 0));
        }
      }
      return;
    }
    case q::Type::q6_k: {
      const float d = h2f(p + 208);
      for (int g = 0; g < 16; ++g) {
        o.sw[g] = d * static_cast<float>(static_cast<std::int8_t>(p[192 + g]));
        o.mw[g] = 0.0f;
      }
      for (int h = 0; ints && h < 2; ++h) {
        const std::uint8_t* ql = p + 64 * h;
        const std::uint8_t* qh = p + 128 + 32 * h;
        for (int l = 0; l < 32; ++l) {
          y[128 * h + l] = i8(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
          y[128 * h + l + 32] = i8(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
          y[128 * h + l + 64] = i8(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
          y[128 * h + l + 96] = i8(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
      }
      return;
    }
    case q::Type::q2_k: {
      const float d = h2f(p + 80), mn = h2f(p + 82);
      for (int g = 0; g < 16; ++g) {
        o.sw[g] = d * static_cast<float>(p[g] & 0xF);
        o.mw[g] = mn * static_cast<float>(p[g] >> 4);
      }
      for (int h = 0; h < 2; ++h)
        for (int j = 0; j < 4; ++j)
          for (int l = 0; l < 32; ++l) y[128 * h + 32 * j + l] = i8((p[16 + 32 * h + l] >> (2 * j)) & 3);
      return;
    }
    case q::Type::q3_k: {
      const float d_all = h2f(p + 108);
      std::uint32_t aux[4];
      std::memcpy(aux, p + 96, 12);
      constexpr std::uint32_t k1 = 0x03030303, k2 = 0x0f0f0f0f;
      const std::uint32_t tmp = aux[2];
      aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
      aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
      aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4);
      aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
      std::int8_t scales[16];
      std::memcpy(scales, aux, 16);
      for (int g = 0; g < 16; ++g) {
        o.sw[g] = d_all * static_cast<float>(scales[g] - 32);
        o.mw[g] = 0.0f;
      }
      for (int h = 0; h < 2; ++h)
        for (int j = 0; j < 4; ++j)
          for (int l = 0; l < 32; ++l)
            y[128 * h + 32 * j + l] = i8(((p[32 + 32 * h + l] >> (2 * j)) & 3) - ((p[l] & (1 << (4 * h + j))) ? 0 : 4));
      return;
    }
    case q::Type::tq2_0: {
      const float d = h2f(p + 64);
      for (int g = 0; g < 16; ++g) { o.sw[g] = d; o.mw[g] = 0.0f; }
      for (int j = 0; j < 2; ++j)
        for (int l = 0; l < 4; ++l)
          for (int m = 0; m < 32; ++m) y[128 * j + 32 * l + m] = i8(((p[32 * j + m] >> (2 * l)) & 3) - 1);
      return;
    }
    case q::Type::tq1_0: {
      const float d = h2f(p + 52);
      for (int g = 0; g < 16; ++g) { o.sw[g] = d; o.mw[g] = 0.0f; }
      static const int pow3[5] = {1, 3, 9, 27, 81};
      auto trit = [](int x) { return ((x & 255) * 3 >> 8) - 1; };
      int i = 0;
      for (int nn = 0; nn < 5; ++nn)
        for (int m = 0; m < 32; ++m) y[i++] = i8(trit(p[m] * pow3[nn]));
      for (int nn = 0; nn < 5; ++nn)
        for (int m = 0; m < 16; ++m) y[i++] = i8(trit(p[32 + m] * pow3[nn]));
      for (int nn = 0; nn < 4; ++nn)
        for (int j = 0; j < 4; ++j) y[i++] = i8(trit(p[48 + j] * pow3[nn]));
      return;
    }
    case q::Type::iq2_xxs: case q::Type::iq2_xs: case q::Type::iq2_s: case q::Type::iq3_xxs: case q::Type::iq3_s: {
      const float d = h2f(p);
      for (int ib = 0; ib < 8; ++ib) {
        std::uint64_t g8[4];
        std::uint8_t sg[4];
        float db[2];
        if (t == q::Type::iq2_xxs) {
          const std::uint8_t* qq = p + 2 + 8 * ib;
          const std::uint32_t aux1 = u32le(qq + 4);
          db[0] = db[1] = d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
          for (int l = 0; l < 4; ++l) { g8[l] = q::iq::kIq2xxsGrid[qq[l]]; sg[l] = q::iq::kSigns[(aux1 >> 7 * l) & 127]; }
        } else if (t == q::Type::iq2_xs) {
          const std::uint8_t sc = p[66 + ib];
          db[0] = d * (0.5f + static_cast<float>(sc & 0xf)) * 0.25f;
          db[1] = d * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
          for (int l = 0; l < 4; ++l) {
            const std::uint16_t qv = u16(p + 2 + 2 * (4 * ib + l));
            g8[l] = q::iq::kIq2xsGrid[qv & 511];
            sg[l] = q::iq::kSigns[qv >> 9];
          }
        } else if (t == q::Type::iq2_s) {
          const std::uint8_t sc = p[74 + ib];
          db[0] = d * (0.5f + static_cast<float>(sc & 0xf)) * 0.25f;
          db[1] = d * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
          for (int l = 0; l < 4; ++l) {
            g8[l] = q::iq::kIq2sGrid[p[2 + 4 * ib + l] | ((p[66 + ib] << (8 - 2 * l)) & 0x300)];
            sg[l] = p[34 + 4 * ib + l];
          }
        } else if (t == q::Type::iq3_xxs) {
          const std::uint32_t aux = u32le(p + 66 + 4 * ib);
          db[0] = db[1] = d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
          const std::uint8_t* qs = p + 2 + 8 * ib;
          for (int l = 0; l < 4; ++l) {
            g8[l] = static_cast<std::uint64_t>(q::iq::kIq3xxsGrid[qs[2 * l]]) | (static_cast<std::uint64_t>(q::iq::kIq3xxsGrid[qs[2 * l + 1]]) << 32);
            sg[l] = q::iq::kSigns[(aux >> 7 * l) & 127];
          }
        } else {  // iq3_s: blocks in pairs share a scale byte; qh per block
          const std::uint8_t sc = p[106 + ib / 2];
          db[0] = db[1] = d * static_cast<float>(1 + 2 * ((ib % 2 == 0) ? (sc & 0xf) : (sc >> 4)));
          const std::uint8_t* qs = p + 2 + 8 * ib;
          const std::uint8_t qh = p[66 + ib];
          for (int l = 0; l < 4; ++l) {
            const std::uint32_t g1 = q::iq::kIq3sGrid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)];
            const std::uint32_t g2 = q::iq::kIq3sGrid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
            g8[l] = static_cast<std::uint64_t>(g1) | (static_cast<std::uint64_t>(g2) << 32);
            sg[l] = p[74 + 4 * ib + l];
          }
        }
        o.sw[2 * ib] = db[0];
        o.sw[2 * ib + 1] = db[1];
        o.mw[2 * ib] = o.mw[2 * ib + 1] = 0.0f;
        for (int l = 0; l < 4; ++l)
          for (int j = 0; j < 8; ++j) {
            const int v = static_cast<int>((g8[l] >> (8 * j)) & 0xFF);
            y[32 * ib + 8 * l + j] = i8(((sg[l] >> j) & 1) ? -v : v);
          }
      }
      return;
    }
    case q::Type::iq1_s: {  // dl * (g + delta) = dl * g - (-dl * delta)
      const float d = h2f(p);
      for (int ib = 0; ib < 8; ++ib) {
        const std::uint16_t qh = u16(p + 34 + 2 * ib);
        const float dl = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
        set32(o, ib, dl, -(dl * ((qh & 0x8000) ? -kIq1Delta : kIq1Delta)));
        for (int l = 0; l < 4; ++l) {
          const std::uint64_t g = q::iq::kIq1sGrid[p[2 + 4 * ib + l] | (((qh >> 3 * l) & 7) << 8)];
          for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = static_cast<std::int8_t>((g >> (8 * j)) & 0xFF);
        }
      }
      return;
    }
    default:
      return;
  }
}

// the integer sums of one 32-block: lane j = values 4j .. 4j + 3 (the AVX2 madd lanes)
void isum_lanes_portable(const std::int8_t* w, const std::int8_t* a, std::int32_t* lane) {
  for (int j = 0; j < 8; ++j) {
    std::int32_t acc = 0;
    for (int i = 0; i < 4; ++i) acc += static_cast<std::int32_t>(w[4 * j + i]) * a[4 * j + i];
    lane[j] = acc;
  }
}

float reduce8(const float* l) {  // (0+4, 1+5, 2+6, 3+7), (0+2, 1+3), 0+1, as reduce_portable's last steps
  float d[4], e[2];
  for (int j = 0; j < 4; ++j) d[j] = l[j] + l[4 + j];
  for (int j = 0; j < 2; ++j) e[j] = d[j] + d[2 + j];
  return e[0] + e[1];
}

// planes in integers: per plane and 32-block, s_p * (s_b * (2 sum_{sign set} q - A_b)), plane-major then block order
float planes_q8_seg(const std::uint8_t* o, int k_stored, int k_use, const ActQ8x32& a, std::int64_t seg) {
  const std::uint8_t* bits = o + 2 * k_stored;
  float acc = 0.0f;
  for (int pl = 0; pl < k_use; ++pl) {
    const float sp = h2f(o + 2 * pl);
    const std::uint8_t* bp = bits + 32 * pl;
    for (int b = 0; b < 8; ++b) {
      const std::int64_t i0 = seg * kSeg + 32 * b;
      std::int32_t set = 0;
      for (int i = 0; i < 32; ++i)
        if ((bp[(32 * b + i) / 8] >> ((32 * b + i) % 8)) & 1) set += a.q[static_cast<std::size_t>(i0 + i)];
      const std::int32_t ab = a.sum16[static_cast<std::size_t>(i0 / 16)] + a.sum16[static_cast<std::size_t>(i0 / 16 + 1)];
      acc += sp * (a.s[static_cast<std::size_t>(i0 / 32)] * static_cast<float>(2 * set - ab));
    }
  }
  return acc;
}

// ------------------------------------------------------------------------------------------------------- AVX2 path
#if defined(PELIAS_KERN_X86)

// 8 bytes (low half of b) as u8 -> fp32
PELIAS_TARGET_AVX2 inline __m256 u8x8(__m128i b) { return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b)); }
PELIAS_TARGET_AVX2 inline __m256 i8x8(__m128i b) { return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)); }

// 32 unsigned 4-bit values (one byte each, 0..15 or 0..31) as d * q - m, written to y[0..32)
PELIAS_TARGET_AVX2 inline void dq32_dm(__m256i v, float d, float m, float* y) {
  const __m256 dv = _mm256_set1_ps(d), mv = _mm256_set1_ps(m);
  const __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
  _mm256_storeu_ps(y + 0, _mm256_sub_ps(_mm256_mul_ps(dv, u8x8(lo)), mv));
  _mm256_storeu_ps(y + 8, _mm256_sub_ps(_mm256_mul_ps(dv, u8x8(_mm_srli_si128(lo, 8))), mv));
  _mm256_storeu_ps(y + 16, _mm256_sub_ps(_mm256_mul_ps(dv, u8x8(hi)), mv));
  _mm256_storeu_ps(y + 24, _mm256_sub_ps(_mm256_mul_ps(dv, u8x8(_mm_srli_si128(hi, 8))), mv));
}

PELIAS_TARGET_AVX2 void deq_q8_0(const std::uint8_t* p, float* y) {  // 8 blocks: d (fp16), 32 x int8; y = q * d
  for (int b = 0; b < 8; ++b, p += 34, y += 32) {
    const __m256 d = _mm256_set1_ps(h2f(p));
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 2));
    const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 18));
    _mm256_storeu_ps(y + 0, _mm256_mul_ps(i8x8(a), d));
    _mm256_storeu_ps(y + 8, _mm256_mul_ps(i8x8(_mm_srli_si128(a, 8)), d));
    _mm256_storeu_ps(y + 16, _mm256_mul_ps(i8x8(c), d));
    _mm256_storeu_ps(y + 24, _mm256_mul_ps(i8x8(_mm_srli_si128(c, 8)), d));
  }
}

PELIAS_TARGET_AVX2 void deq_q4_k(const std::uint8_t* p, float* y) {  // d, dmin, scales[12], qs[128]
  const float d = h2f(p), mn = h2f(p + 2);
  const std::uint8_t* sc = p + 4;
  const std::uint8_t* qs = p + 16;
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  for (int j = 0; j < 4; ++j, qs += 32, y += 64) {
    int s1, m1, s2, m2;
    scale_min_k4(2 * j, sc, &s1, &m1);
    scale_min_k4(2 * j + 1, sc, &s2, &m2);
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
    dq32_dm(_mm256_and_si256(v, m4), d * static_cast<float>(s1), mn * static_cast<float>(m1), y);
    dq32_dm(_mm256_and_si256(_mm256_srli_epi16(v, 4), m4), d * static_cast<float>(s2), mn * static_cast<float>(m2), y + 32);
  }
}

PELIAS_TARGET_AVX2 void deq_q5_k(const std::uint8_t* p, float* y) {  // d, dmin, scales[12], qh[32], ql[128]
  const float d = h2f(p), mn = h2f(p + 2);
  const std::uint8_t* sc = p + 4;
  const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 16));
  const std::uint8_t* ql = p + 48;
  const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
  for (int j = 0; j < 4; ++j, ql += 32, y += 64) {
    int s1, m1, s2, m2;
    scale_min_k4(2 * j, sc, &s1, &m1);
    scale_min_k4(2 * j + 1, sc, &s2, &m2);
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
    // bit 2j (low nibbles) and bit 2j + 1 (high nibbles) of each qh byte, as +16
    const __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * j), one), 4);
    const __m256i h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * j + 1), one), 4);
    dq32_dm(_mm256_add_epi8(_mm256_and_si256(v, m4), h1), d * static_cast<float>(s1), mn * static_cast<float>(m1), y);
    dq32_dm(_mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(v, 4), m4), h2), d * static_cast<float>(s2),
            mn * static_cast<float>(m2), y + 32);
  }
}

// 16 signed bytes (low 128 bits of v) as ds * q, for one Q6_K 16-value group
PELIAS_TARGET_AVX2 inline void dq16_s(__m128i v, float ds, float* y) {
  const __m256 dv = _mm256_set1_ps(ds);
  _mm256_storeu_ps(y + 0, _mm256_mul_ps(dv, i8x8(v)));
  _mm256_storeu_ps(y + 8, _mm256_mul_ps(dv, i8x8(_mm_srli_si128(v, 8))));
}

// bits sh, sh + 1 of each qh byte, moved to bits 4-5
PELIAS_TARGET_AVX2 inline __m256i q6_hi2(__m256i hb, int sh, __m256i m3) {
  return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, sh), m3), 4);
}

PELIAS_TARGET_AVX2 void deq_q6_k(const std::uint8_t* p, float* y) {  // ql[128], qh[64], scales[16] (int8), d (fp16)
  const float d = h2f(p + 208);
  std::int8_t scv[16];
  std::memcpy(scv, p + 192, 16);
  const std::uint8_t* ql = p;
  const std::uint8_t* qh = p + 128;
  const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(3), o32 = _mm256_set1_epi8(32);
  for (int h = 0; h < 2; ++h, ql += 64, qh += 32, y += 128) {
    const std::int8_t* sc = scv + 8 * h;
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));        // ql[l]
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));   // ql[l + 32]
    const __m256i hb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
    const __m256i q1 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(a, m4), q6_hi2(hb, 0, m3)), o32);
    const __m256i q2 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(b, m4), q6_hi2(hb, 2, m3)), o32);
    const __m256i q3 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a, 4), m4), q6_hi2(hb, 4, m3)), o32);
    const __m256i q4 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b, 4), m4), q6_hi2(hb, 6, m3)), o32);
    const __m256i qv[4] = {q1, q2, q3, q4};
    for (int r = 0; r < 4; ++r) {  // value l of quarter r goes to y[32 r + l]; scale sc[2 r + l / 16]
      const float d0 = d * static_cast<float>(sc[2 * r]), d1 = d * static_cast<float>(sc[2 * r + 1]);
      dq16_s(_mm256_castsi256_si128(qv[r]), d0, y + 32 * r);
      dq16_s(_mm256_extracti128_si256(qv[r], 1), d1, y + 32 * r + 16);
    }
  }
}

PELIAS_TARGET_AVX2 void deq_iq4_xs(const std::uint8_t* p, float* y) {  // d, scales_h (u16), scales_l[4], qs[128]
  const float d = h2f(p);
  const std::uint16_t sh = u16(p + 2);
  const std::uint8_t* sl = p + 4;
  const std::uint8_t* qs = p + 8;
  const __m128i tab = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kIq4));
  const __m128i m4 = _mm_set1_epi8(0x0F);
  for (int ib = 0; ib < 8; ++ib, qs += 16, y += 32) {
    const int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
    const float dl = d * static_cast<float>(ls - 32);
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    dq16_s(_mm_shuffle_epi8(tab, _mm_and_si128(v, m4)), dl, y);                              // y[j], j = 0..15
    dq16_s(_mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16(v, 4), m4)), dl, y + 16);      // y[j + 16]
  }
}

// Q4_TG / Q4_TG2: d (fp16), 8 six-bit levels (6 bytes), 8 tag bytes (2 bits per 8-value group), 128 bytes of nibbles
// (value i in byte 16 + i / 2, low nibble for even i). y = s_j * grid[tag][code], s_j = (d * lv_j) / 63.
PELIAS_TARGET_AVX2 void deq_q4_tg(const std::uint8_t* p, float* y, int version) {
  const float d = h2f(p);
  std::uint64_t packed = 0;
  for (int b = 0; b < 6; ++b) packed |= static_cast<std::uint64_t>(p[2 + b]) << (8 * b);
  __m256 glo[4], ghi[4];
  for (int t = 0; t < 4; ++t) {
    const float* g = q::q4_tg_grid(t, version);
    glo[t] = _mm256_loadu_ps(g);
    ghi[t] = _mm256_loadu_ps(g + 8);
  }
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m256i seven = _mm256_set1_epi32(7);
  for (int j = 0; j < 8; ++j) {
    const __m256 s = _mm256_set1_ps(q::q8_z_scale(d, static_cast<int>((packed >> (6 * j)) & 63u)));
    for (int g = 0; g < 4; ++g) {
      const int t = (p[8 + j] >> (2 * g)) & 3;
      std::int32_t w4;
      std::memcpy(&w4, p + 16 + (32 * j + 8 * g) / 2, 4);
      const __m128i x = _mm_cvtsi32_si128(w4);
      const __m128i codes = _mm_unpacklo_epi8(_mm_and_si128(x, m4), _mm_and_si128(_mm_srli_epi16(x, 4), m4));
      const __m256i idx = _mm256_cvtepu8_epi32(codes);
      const __m256 lo = _mm256_permutevar8x32_ps(glo[t], idx), hi = _mm256_permutevar8x32_ps(ghi[t], idx);
      const __m256 v = _mm256_blendv_ps(lo, hi, _mm256_castsi256_ps(_mm256_cmpgt_epi32(idx, seven)));
      _mm256_storeu_ps(y + 32 * j + 8 * g, _mm256_mul_ps(s, v));
    }
  }
}

// ---- the 32-value block types: one segment = eight blocks

// 32 signed bytes (v) as float(v) * d, or float(v) * d + m when add_m, to y[0..32)
PELIAS_TARGET_AVX2 inline void dq32_s(__m256i v, float d, float m, bool add_m, float* y) {
  const __m256 dv = _mm256_set1_ps(d), mv = _mm256_set1_ps(m);
  const __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
  const __m128i parts[4] = {lo, _mm_srli_si128(lo, 8), hi, _mm_srli_si128(hi, 8)};
  for (int r = 0; r < 4; ++r) {
    const __m256 p = _mm256_mul_ps(i8x8(parts[r]), dv);
    _mm256_storeu_ps(y + 8 * r, add_m ? _mm256_add_ps(p, mv) : p);
  }
}

// 16 packed bytes -> 32 bytes: the low nibbles (values 0..15) in lane 0, the high nibbles (values 16..31) in lane 1
PELIAS_TARGET_AVX2 inline __m256i nibbles32(const std::uint8_t* qs) {
  const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
  const __m128i m4 = _mm_set1_epi8(0x0F);
  return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m4), _mm_and_si128(v, m4));
}

// the fifth bits of Q5_0 / Q5_1: bit j of qh as 0x10 in byte j (j = 0..31)
PELIAS_TARGET_AVX2 inline __m256i q5_high(std::uint32_t qh) {
  const __m256i b = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(qh)),
                                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
  const __m256i bit = _mm256_set1_epi64x(static_cast<long long>(0x8040201008040201ull));
  return _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(b, bit), bit), _mm256_set1_epi8(0x10));
}


PELIAS_TARGET_AVX2 void deq_q4_0(const std::uint8_t* p, float* y) {  // d, qs[16]; (q - 8) * d
  for (int b = 0; b < 8; ++b, p += 18, y += 32) dq32_s(_mm256_sub_epi8(nibbles32(p + 2), _mm256_set1_epi8(8)), h2f(p), 0.0f, false, y);
}
PELIAS_TARGET_AVX2 void deq_q4_1(const std::uint8_t* p, float* y) {  // d, m, qs[16]; q * d + m
  for (int b = 0; b < 8; ++b, p += 20, y += 32) dq32_s(nibbles32(p + 4), h2f(p), h2f(p + 2), true, y);
}
PELIAS_TARGET_AVX2 void deq_q5_0(const std::uint8_t* p, float* y) {  // d, qh[4], qs[16]; ((q | h) - 16) * d
  for (int b = 0; b < 8; ++b, p += 22, y += 32)
    dq32_s(_mm256_sub_epi8(_mm256_or_si256(nibbles32(p + 6), q5_high(u32le(p + 2))), _mm256_set1_epi8(16)), h2f(p), 0.0f, false, y);
}
PELIAS_TARGET_AVX2 void deq_q5_1(const std::uint8_t* p, float* y) {  // d, m, qh[4], qs[16]; (q | h) * d + m
  for (int b = 0; b < 8; ++b, p += 24, y += 32)
    dq32_s(_mm256_or_si256(nibbles32(p + 8), q5_high(u32le(p + 4))), h2f(p), h2f(p + 2), true, y);
}
PELIAS_TARGET_AVX2 void deq_iq4_nl(const std::uint8_t* p, float* y) {  // d, qs[16]; d * table[q]
  const __m256i tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kIq4)));
  for (int b = 0; b < 8; ++b, p += 18, y += 32) {
    const float d = h2f(p);
    const __m256i v = _mm256_shuffle_epi8(tab, nibbles32(p + 2));
    dq16_s(_mm256_castsi256_si128(v), d, y);
    dq16_s(_mm256_extracti128_si256(v, 1), d, y + 16);
  }
}
// NVFP4 (GGUF type 40), one 256-segment = 4 blocks of 36 bytes: 4 UE4M3 scales, then per 16-value sub-block 8 bytes whose
// low nibbles are values 0-7 and high nibbles values 8-15; value = kFp4[code] x scale (the scale's decode halved)
PELIAS_TARGET_AVX2 void deq_nvfp4(const std::uint8_t* p, float* y) {
  const __m128i tab = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kFp4)), m4 = _mm_set1_epi8(0x0F);
  for (int b = 0; b < 4; ++b, p += 36)
    for (int s = 0; s < 4; ++s, y += 16) {
      const __m128i q = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 4 + 8 * s));
      const __m128i v = _mm_shuffle_epi8(tab, _mm_unpacklo_epi64(_mm_and_si128(q, m4), _mm_and_si128(_mm_srli_epi16(q, 4), m4)));
      const __m256 d = _mm256_set1_ps(q::ue4m3_to_fp32(p[s]));
      _mm256_storeu_ps(y, _mm256_mul_ps(i8x8(v), d));
      _mm256_storeu_ps(y + 8, _mm256_mul_ps(i8x8(_mm_srli_si128(v, 8)), d));
    }
}

PELIAS_TARGET_AVX2 void deq_mxfp4(const std::uint8_t* p, float* y) {  // e (E8M0), qs[16]; float(fp4[q]) * d
  const __m256i tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kFp4)));
  for (int b = 0; b < 8; ++b, p += 17, y += 32) dq32_s(_mm256_shuffle_epi8(tab, nibbles32(p + 1)), e8m0_half(p[0]), 0.0f, false, y);
}

PELIAS_TARGET_AVX2 void deq_q8_z(const std::uint8_t* p, float* y) {  // d, 8 six-bit levels, 256 x int8; float(q) * s_j
  const float d = h2f(p);
  std::uint64_t packed = 0;
  for (int b = 0; b < 6; ++b) packed |= static_cast<std::uint64_t>(p[2 + b]) << (8 * b);
  for (int j = 0; j < 8; ++j)
    dq32_s(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8 + 32 * j)), q::q8_z_scale(d, static_cast<int>((packed >> (6 * j)) & 63u)),
           0.0f, false, y + 32 * j);
}

// ---- Q2_K and Q3_K: 16-value groups

// 16 bytes (low or high half of v) as d * float(q) - m (Q2_K) to y[0..16)
PELIAS_TARGET_AVX2 inline void dq16_dm(__m128i v, float d, float m, float* y) {
  const __m256 dv = _mm256_set1_ps(d), mv = _mm256_set1_ps(m);
  _mm256_storeu_ps(y + 0, _mm256_sub_ps(_mm256_mul_ps(dv, u8x8(v)), mv));
  _mm256_storeu_ps(y + 8, _mm256_sub_ps(_mm256_mul_ps(dv, u8x8(_mm_srli_si128(v, 8))), mv));
}

PELIAS_TARGET_AVX2 void deq_q2_k(const std::uint8_t* p, float* y) {  // scales[16], qs[64], d, dmin
  const float d = h2f(p + 80), mn = h2f(p + 82);
  const std::uint8_t* sc = p;
  const __m256i m3 = _mm256_set1_epi8(3);
  int is = 0;
  for (int h = 0; h < 2; ++h) {
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 16 + 32 * h));
    for (int j = 0; j < 4; ++j, y += 32) {
      const __m256i v = _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m3);
      const std::uint8_t s0 = sc[is++], s1 = sc[is++];
      dq16_dm(_mm256_castsi256_si128(v), d * static_cast<float>(s0 & 0xF), mn * static_cast<float>(s0 >> 4), y);
      dq16_dm(_mm256_extracti128_si256(v, 1), d * static_cast<float>(s1 & 0xF), mn * static_cast<float>(s1 >> 4), y + 16);
    }
  }
}

PELIAS_TARGET_AVX2 void deq_q3_k(const std::uint8_t* p, float* y) {  // hmask[32], qs[64], scales[12], d
  const float d_all = h2f(p + 108);
  std::uint32_t aux[4];
  std::memcpy(aux, p + 96, 12);
  constexpr std::uint32_t k1 = 0x03030303, k2 = 0x0f0f0f0f;
  const std::uint32_t tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
  aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
  aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4);
  aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
  std::int8_t scales[16];
  std::memcpy(scales, aux, 16);
  const __m256i hm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  const __m256i m3 = _mm256_set1_epi8(3), four = _mm256_set1_epi8(4), zero = _mm256_setzero_si256();
  int is = 0, bit = 0;
  for (int h = 0; h < 2; ++h) {
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32 + 32 * h));
    for (int j = 0; j < 4; ++j, ++bit, y += 32) {
      const __m256i lo2 = _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m3);
      const __m256i clear = _mm256_cmpeq_epi8(_mm256_and_si256(hm, _mm256_set1_epi8(static_cast<char>(1 << bit))), zero);
      const __m256i v = _mm256_sub_epi8(lo2, _mm256_and_si256(clear, four));
      dq16_s(_mm256_castsi256_si128(v), d_all * static_cast<float>(scales[is] - 32), y);
      dq16_s(_mm256_extracti128_si256(v, 1), d_all * static_cast<float>(scales[is + 1] - 32), y + 16);
      is += 2;
    }
  }
}

// ---- plain floats
PELIAS_TARGET_AVX2 void deq_f16(const std::uint8_t* p, float* y) {
  for (int i = 0; i < kSeg; i += 8) _mm256_storeu_ps(y + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 2 * i))));
}
PELIAS_TARGET_AVX2 void deq_bf16(const std::uint8_t* p, float* y) {
  for (int i = 0; i < kSeg; i += 8)
    _mm256_storeu_ps(y + i, _mm256_castsi256_ps(_mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 2 * i))), 16)));
}

// ---- the grid types (IQ1 / IQ2 / IQ3): 8 values per grid entry, with sign bits

// the 8 bytes of a grid entry as floats (unsigned, or signed for the IQ1 grid)
PELIAS_TARGET_AVX2 inline __m256 grid8u(std::uint64_t g) {
  return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(g))));
}
PELIAS_TARGET_AVX2 inline __m256 grid8s(std::uint64_t g) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(static_cast<long long>(g))));
}
PELIAS_TARGET_AVX2 inline __m256 grid4x2(std::uint32_t g1, std::uint32_t g2) {  // two 4-byte entries (IQ3)
  return grid8u(static_cast<std::uint64_t>(g1) | (static_cast<std::uint64_t>(g2) << 32));
}
// bit j of `signs` set: value j negative. x * (+/-1) is exact, so flipping the sign bit equals the reference's product.
PELIAS_TARGET_AVX2 inline __m256 apply_signs(__m256 v, std::uint8_t signs) {
  const __m256i bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  const __m256i set = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(signs), bit), bit);
  return _mm256_xor_ps(v, _mm256_and_ps(_mm256_castsi256_ps(set), _mm256_set1_ps(-0.0f)));
}

PELIAS_TARGET_AVX2 void deq_iq2_xxs(const std::uint8_t* p, float* y) {  // d, 8 bytes per 32: 4 grid indices, u32 (4 x 7 sign bits, 4-bit scale)
  const float d = h2f(p);
  for (int ib = 0; ib < 8; ++ib) {
    const std::uint8_t* qq = p + 2 + 8 * ib;
    const std::uint32_t aux1 = u32le(qq + 4);
    const __m256 db = _mm256_set1_ps(d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f);
    for (int l = 0; l < 4; ++l, y += 8)
      _mm256_storeu_ps(y, apply_signs(_mm256_mul_ps(db, grid8u(q::iq::kIq2xxsGrid[qq[l]])), q::iq::kSigns[(aux1 >> 7 * l) & 127]));
  }
}

PELIAS_TARGET_AVX2 void deq_iq2_xs(const std::uint8_t* p, float* y) {  // d, qs u16[32], scales[8]
  const float d = h2f(p);
  const std::uint8_t* sc = p + 66;
  for (int ib = 0; ib < 8; ++ib) {
    const float db[2] = {d * (0.5f + static_cast<float>(sc[ib] & 0xf)) * 0.25f, d * (0.5f + static_cast<float>(sc[ib] >> 4)) * 0.25f};
    for (int l = 0; l < 4; ++l, y += 8) {
      const std::uint16_t qv = u16(p + 2 + 2 * (4 * ib + l));
      _mm256_storeu_ps(y, apply_signs(_mm256_mul_ps(_mm256_set1_ps(db[l / 2]), grid8u(q::iq::kIq2xsGrid[qv & 511])), q::iq::kSigns[qv >> 9]));
    }
  }
}

PELIAS_TARGET_AVX2 void deq_iq2_s(const std::uint8_t* p, float* y) {  // d, qs[32] + signs[32], qh[8], scales[8]
  const float d = h2f(p);
  const std::uint8_t* qs = p + 2;
  const std::uint8_t* signs = qs + 32;
  const std::uint8_t* qh = p + 66;
  const std::uint8_t* sc = p + 74;
  for (int ib = 0; ib < 8; ++ib, qs += 4, signs += 4) {
    const float db[2] = {d * (0.5f + static_cast<float>(sc[ib] & 0xf)) * 0.25f, d * (0.5f + static_cast<float>(sc[ib] >> 4)) * 0.25f};
    for (int l = 0; l < 4; ++l, y += 8) {
      const std::uint64_t g = q::iq::kIq2sGrid[qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300)];
      _mm256_storeu_ps(y, apply_signs(_mm256_mul_ps(_mm256_set1_ps(db[l / 2]), grid8u(g)), signs[l]));
    }
  }
}

PELIAS_TARGET_AVX2 void deq_iq3_xxs(const std::uint8_t* p, float* y) {  // d, qs[64] grid indices, then 32 bytes of scales and signs
  const float d = h2f(p);
  const std::uint8_t* qs = p + 2;
  const std::uint8_t* ss = p + 2 + 64;
  for (int ib = 0; ib < 8; ++ib, qs += 8) {
    const std::uint32_t aux = u32le(ss + 4 * ib);
    const __m256 db = _mm256_set1_ps(d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f);
    for (int l = 0; l < 4; ++l, y += 8)
      _mm256_storeu_ps(y, apply_signs(_mm256_mul_ps(db, grid4x2(q::iq::kIq3xxsGrid[qs[2 * l]], q::iq::kIq3xxsGrid[qs[2 * l + 1]])),
                                      q::iq::kSigns[(aux >> 7 * l) & 127]));
  }
}

PELIAS_TARGET_AVX2 void deq_iq3_s(const std::uint8_t* p, float* y) {  // d, qs[64], qh[8], signs[32], scales[4]
  const float d = h2f(p);
  const std::uint8_t* qs = p + 2;
  const std::uint8_t* qh = p + 66;
  const std::uint8_t* signs = p + 74;
  const std::uint8_t* sc = p + 106;
  for (int ib = 0; ib < 8; ib += 2, qh += 2) {
    const float dbs[2] = {d * static_cast<float>(1 + 2 * (sc[ib / 2] & 0xf)), d * static_cast<float>(1 + 2 * (sc[ib / 2] >> 4))};
    for (int half = 0; half < 2; ++half, qs += 8, signs += 4) {
      const __m256 db = _mm256_set1_ps(dbs[half]);
      for (int l = 0; l < 4; ++l, y += 8) {
        const std::uint32_t g1 = q::iq::kIq3sGrid[qs[2 * l + 0] | ((qh[half] << (8 - 2 * l)) & 256)];
        const std::uint32_t g2 = q::iq::kIq3sGrid[qs[2 * l + 1] | ((qh[half] << (7 - 2 * l)) & 256)];
        _mm256_storeu_ps(y, apply_signs(_mm256_mul_ps(db, grid4x2(g1, g2)), signs[l]));
      }
    }
  }
}


PELIAS_TARGET_AVX2 void deq_iq1_s(const std::uint8_t* p, float* y) {  // d, qs[32], qh u16[8]; dl * (grid + delta)
  const float d = h2f(p);
  const std::uint8_t* qs = p + 2;
  for (int ib = 0; ib < 8; ++ib, qs += 4) {
    const std::uint16_t qh = u16(p + 34 + 2 * ib);
    const __m256 dl = _mm256_set1_ps(d * static_cast<float>(2 * ((qh >> 12) & 7) + 1));
    const __m256 delta = _mm256_set1_ps((qh & 0x8000) ? -kIq1Delta : kIq1Delta);
    for (int l = 0; l < 4; ++l, y += 8)
      _mm256_storeu_ps(y, _mm256_mul_ps(dl, _mm256_add_ps(grid8s(q::iq::kIq1sGrid[qs[l] | (((qh >> 3 * l) & 7) << 8)]), delta)));
  }
}

PELIAS_TARGET_AVX2 void deq_iq1_m(const std::uint8_t* p, float* y) {  // qs[32], qh[16], scales[8] (fp16 scale in the top nibbles)
  const std::uint8_t* qs = p;
  const std::uint8_t* qh = p + 32;
  std::uint16_t sc[4];
  for (int k = 0; k < 4; ++k) sc[k] = u16(p + 48 + 2 * k);
  const std::uint16_t s16 = static_cast<std::uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
  const float d = q::fp16_to_fp32(s16);
  for (int ib = 0; ib < 8; ++ib, qs += 4, qh += 2) {
    const float dl1 = d * static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
    const float dl2 = d * static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
    const int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700), qs[2] | ((qh[1] << 8) & 0x700), qs[3] | ((qh[1] << 4) & 0x700)};
    const float delta[4] = {(qh[0] & 0x08) ? -kIq1Delta : kIq1Delta, (qh[0] & 0x80) ? -kIq1Delta : kIq1Delta,
                            (qh[1] & 0x08) ? -kIq1Delta : kIq1Delta, (qh[1] & 0x80) ? -kIq1Delta : kIq1Delta};
    for (int l = 0; l < 4; ++l, y += 8)
      _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_set1_ps(l < 2 ? dl1 : dl2),
                                        _mm256_add_ps(grid8s(q::iq::kIq1sGrid[idx[l]]), _mm256_set1_ps(delta[l]))));
  }
}

// ---- ternary (TQ1_0, TQ2_0): float(q - 1) * d with q in 0..2

// I2_S, one 256-segment = two 128-blocks of 32 bytes: byte j holds values j, 32+j, 64+j, 96+j at shifts 6, 4, 2, 0;
// value = code - 1 (the tensor's scale is applied outside the row, as quant::dequantize_i2_s leaves it)
PELIAS_TARGET_AVX2 void deq_i2_s(const std::uint8_t* p, float* y) {
  const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
  for (int b = 0; b < 2; ++b) {
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32 * b));
    for (int g = 0; g < 4; ++g, y += 32) dq32_s(_mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(q, 6 - 2 * g), m3), one), 1.0f, 0.0f, false, y);
  }
}

PELIAS_TARGET_AVX2 void deq_tq2_0(const std::uint8_t* p, float* y) {  // qs[64] (2 bits each), d
  const float d = h2f(p + 64);
  const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
  for (int j = 0; j < 64; j += 32) {
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + j));
    for (int l = 0; l < 4; ++l, y += 32) dq32_s(_mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(q, 2 * l), m3), one), d, 0.0f, false, y);
  }
}

// TQ1_0: 5 trits per byte (qs) and 4 per byte (qh), read as trit(x) = (x * 3) >> 8 of x = (q * 3^k) mod 256
PELIAS_TARGET_AVX2 inline __m256i tq1_trits16(__m256i q16, int pw) {  // 16 bytes widened to u16: (((q * pw) & 255) * 3 >> 8) - 1
  const __m256i v = _mm256_and_si256(_mm256_mullo_epi16(q16, _mm256_set1_epi16(static_cast<short>(pw))), _mm256_set1_epi16(255));
  return _mm256_sub_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(v, _mm256_set1_epi16(3)), 8), _mm256_set1_epi16(1));
}
PELIAS_TARGET_AVX2 inline void tq1_store16(__m256i t16, float d, float* y) {  // 16 int16 trits as float(t) * d
  const __m256 dv = _mm256_set1_ps(d);
  _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(t16))), dv));
  _mm256_storeu_ps(y + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(t16, 1))), dv));
}
PELIAS_TARGET_AVX2 void deq_tq1_0(const std::uint8_t* p, float* y) {  // qs[48], qh[4], d
  const float d = h2f(p + 52);
  static const int pow3[5] = {1, 3, 9, 27, 81};
  const __m256i a0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));       // qs[0..16)
  const __m256i a1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16)));  // qs[16..32)
  const __m256i b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32)));  // qs[32..48)
  for (int nn = 0; nn < 5; ++nn, y += 32) {
    tq1_store16(tq1_trits16(a0, pow3[nn]), d, y);
    tq1_store16(tq1_trits16(a1, pow3[nn]), d, y + 16);
  }
  for (int nn = 0; nn < 5; ++nn, y += 16) tq1_store16(tq1_trits16(b0, pow3[nn]), d, y);
  for (int nn = 0; nn < 4; ++nn)
    for (int j = 0; j < 4; ++j) {
      const std::uint8_t x = static_cast<std::uint8_t>(p[48 + j] * pow3[nn]);
      *y++ = static_cast<float>(static_cast<int>((static_cast<std::uint16_t>(x) * 3) >> 8) - 1) * d;
    }
}

// ---- Q7_Z / Q7_ZT (Pelias research types): eight 7-bit values per 7 bytes

PELIAS_TARGET_AVX2 inline __m256i unpack7(const std::uint8_t* g) {  // the 8 values of one group as int32 (0..127)
  // value k sits at bits 7k..7k+6 of the 7 bytes: values 0-3 from bytes 0-3, values 4-7 from bytes 3-6 (bit 24 on), so
  // exactly the group's 7 bytes are read
  std::uint32_t lo, hi;
  std::memcpy(&lo, g, 4);
  std::memcpy(&hi, g + 3, 4);
  const __m256i v = _mm256_setr_epi32(static_cast<int>(lo), static_cast<int>(lo), static_cast<int>(lo), static_cast<int>(lo), static_cast<int>(hi),
                                      static_cast<int>(hi), static_cast<int>(hi), static_cast<int>(hi));
  return _mm256_and_si256(_mm256_srlv_epi32(v, _mm256_setr_epi32(0, 7, 14, 21, 4, 11, 18, 25)), _mm256_set1_epi32(127));
}

PELIAS_TARGET_AVX2 void deq_q7_z(const std::uint8_t* p, float* y) {  // d, 8 levels, 32 groups of 7 bytes; float(u - 64) * s
  const float d = h2f(p);
  std::uint64_t packed = 0;
  for (int b = 0; b < 6; ++b) packed |= static_cast<std::uint64_t>(p[2 + b]) << (8 * b);
  const __m256i o64 = _mm256_set1_epi32(64);
  for (int j = 0; j < 8; ++j) {
    const __m256 s = _mm256_set1_ps(q::q7_z_scale(d, static_cast<int>((packed >> (6 * j)) & 63u)));
    for (int g = 4 * j; g < 4 * j + 4; ++g, y += 8)
      _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(unpack7(p + 8 + 7 * g), o64)), s));
  }
}

PELIAS_TARGET_AVX2 void deq_q7_zt(const std::uint8_t* p, float* y) {  // d, 8 levels, gate mask, 32 groups of 7 bytes, H words
  const float d = h2f(p);
  std::uint64_t packed = 0;
  for (int b = 0; b < 6; ++b) packed |= static_cast<std::uint64_t>(p[2 + b]) << (8 * b);
  const std::uint8_t mask = p[8];
  const __m256i o127 = _mm256_set1_epi32(127), one = _mm256_set1_epi32(1), lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  int hslot = 0;
  for (int j = 0; j < 8; ++j) {
    const __m256 s = _mm256_set1_ps(q::q8_z_scale(d, static_cast<int>((packed >> (6 * j)) & 63u)));
    const bool gated = ((mask >> j) & 1u) && hslot < 2;  // two H words per block, as the reference reads them
    std::uint32_t hb = 0;
    if (gated) hb = u32le(p + 233 + 4 * hslot++);
    for (int g = 0; g < 4; ++g, y += 8) {
      __m256i qv = _mm256_sub_epi32(_mm256_slli_epi32(unpack7(p + 9 + 7 * (4 * j + g)), 1), o127);  // 2 b - 127
      if (gated)
        qv = _mm256_add_epi32(qv, _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(hb)), _mm256_add_epi32(lane, _mm256_set1_epi32(8 * g))), one));
      _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_cvtepi32_ps(qv), s));
    }
  }
}

// Q4_TG / Q4_TG2 without planes: each value s_j * grid[tag][code] (the same fp32 product as the dequantizer, here taken
// from the four grids scaled once per 32-block) is fused into its lane's multiply-add, in value order, so the sums equal
// the buffered path's bit for bit.
// v45 (Fable review 03:3x): q::q8_z_scale lives in another file (a call per 32 values, no LTO); the same two operations in the
// same order, inline here: (d * lv) / 63, bit for bit.
inline float z_scale_local(float d, int lv) { return (d * static_cast<float>(lv)) / 63.0f; }
// v45b (author 17:21): the eight scales of a 256-value segment from one vector multiply and one vector division. IEEE
// division and multiplication are exact per lane, so each lane equals z_scale_local (checked for every fp16 d and level).
// The six level bytes come from one 8-byte load (bytes 2..9 of the segment; the top 16 bits masked off).
PELIAS_TARGET_AVX2 inline void tg_scales8(const std::uint8_t* p, float d, float* sc) {
  std::uint64_t packed = 0;
  std::memcpy(&packed, p + 2, 8);
  packed &= 0xFFFFFFFFFFFFull;
  const __m256i lv = _mm256_setr_epi32(static_cast<int>(packed & 63u), static_cast<int>((packed >> 6) & 63u), static_cast<int>((packed >> 12) & 63u),
                                       static_cast<int>((packed >> 18) & 63u), static_cast<int>((packed >> 24) & 63u), static_cast<int>((packed >> 30) & 63u),
                                       static_cast<int>((packed >> 36) & 63u), static_cast<int>((packed >> 42) & 63u));
  _mm256_storeu_ps(sc, _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(lv)), _mm256_set1_ps(63.0f)));
}
// v45b: the Q4_TG grids' addresses, fetched from quant once (q4_tg_grid lives in another file: no call per segment)
inline const float* tg_grid(int t, int version) {
  static const float* const g[2][4] = {{q::q4_tg_grid(0, 1), q::q4_tg_grid(1, 1), q::q4_tg_grid(2, 1), q::q4_tg_grid(3, 1)},
                                       {q::q4_tg_grid(0, 2), q::q4_tg_grid(1, 2), q::q4_tg_grid(2, 2), q::q4_tg_grid(3, 2)}};
  return g[version == 2 ? 1 : 0][t & 3];
}

PELIAS_TARGET_AVX2 void fused_q4_tg_seg(const std::uint8_t* p, int version, const float* const* xs, int nt, __m256 (*acc)[4]) {
  const float d = h2f(p);
  alignas(32) float sc[8];
  tg_scales8(p, d, sc);
  __m256 glo[4], ghi[4];
  for (int t = 0; t < 4; ++t) {
    const float* g = tg_grid(t, version);
    glo[t] = _mm256_loadu_ps(g);
    ghi[t] = _mm256_loadu_ps(g + 8);
  }
  const __m128i m4 = _mm_set1_epi8(0x0F);
  for (int j = 0; j < 8; ++j) {
    const __m256 sv = _mm256_set1_ps(sc[j]);
    const std::uint8_t tags = p[8 + j];
    // the block's 32 codes in value order (value 2i: low nibble of byte i, value 2i + 1: high nibble)
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16 + 16 * j));
    const __m128i lo = _mm_and_si128(v, m4), hi = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
    const __m128i c0 = _mm_unpacklo_epi8(lo, hi), c1 = _mm_unpackhi_epi8(lo, hi);
    const __m128i cg[4] = {c0, _mm_srli_si128(c0, 8), c1, _mm_srli_si128(c1, 8)};
    for (int g = 0; g < 4; ++g) {
      const int t = (tags >> (2 * g)) & 3;
      const __m256i idx = _mm256_cvtepu8_epi32(cg[g]);
      const __m256 lo8 = _mm256_permutevar8x32_ps(glo[t], idx), hi8 = _mm256_permutevar8x32_ps(ghi[t], idx);
      const __m256 w = _mm256_mul_ps(sv, _mm256_blendv_ps(lo8, hi8, _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28))));
      const int at = 32 * j + 8 * g;
      for (int tt = 0; tt < nt; ++tt) acc[tt][g] = _mm256_fmadd_ps(w, _mm256_loadu_ps(xs[tt] + at), acc[tt][g]);
    }
  }
}


// v40 (author 02:30): two rows of Q4_TG / Q4_TG2 against one float vector, the grids loaded once by the caller and each x
// load shared. Per row the same products in the same order as fused_q4_tg_seg with nt = 1 (bit for bit).
PELIAS_TARGET_AVX2 void fused_q4_tg_seg2(const std::uint8_t* p0, const std::uint8_t* p1, const __m256* glo, const __m256* ghi, const float* x,
                                         __m256* a0, __m256* a1) {
  alignas(32) float sc0[8], sc1[8];
  tg_scales8(p0, h2f(p0), sc0);
  tg_scales8(p1, h2f(p1), sc1);
  const __m128i m4 = _mm_set1_epi8(0x0F);
  for (int j = 0; j < 8; ++j) {
    const __m256 sv0 = _mm256_set1_ps(sc0[j]);
    const __m256 sv1 = _mm256_set1_ps(sc1[j]);
    const std::uint8_t tg0 = p0[8 + j], tg1 = p1[8 + j];
    const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p0 + 16 + 16 * j));
    const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + 16 + 16 * j));
    const __m128i lo0 = _mm_and_si128(v0, m4), hi0 = _mm_and_si128(_mm_srli_epi16(v0, 4), m4);
    const __m128i lo1 = _mm_and_si128(v1, m4), hi1 = _mm_and_si128(_mm_srli_epi16(v1, 4), m4);
    const __m128i c00 = _mm_unpacklo_epi8(lo0, hi0), c01 = _mm_unpackhi_epi8(lo0, hi0);
    const __m128i c10 = _mm_unpacklo_epi8(lo1, hi1), c11 = _mm_unpackhi_epi8(lo1, hi1);
    const __m128i cg0[4] = {c00, _mm_srli_si128(c00, 8), c01, _mm_srli_si128(c01, 8)};
    const __m128i cg1[4] = {c10, _mm_srli_si128(c10, 8), c11, _mm_srli_si128(c11, 8)};
    for (int g = 0; g < 4; ++g) {
      const __m256 xv = _mm256_loadu_ps(x + 32 * j + 8 * g);
      const int t0 = (tg0 >> (2 * g)) & 3, t1 = (tg1 >> (2 * g)) & 3;
      const __m256i i0 = _mm256_cvtepu8_epi32(cg0[g]), i1 = _mm256_cvtepu8_epi32(cg1[g]);
      const __m256 w0 = _mm256_mul_ps(sv0, _mm256_blendv_ps(_mm256_permutevar8x32_ps(glo[t0], i0), _mm256_permutevar8x32_ps(ghi[t0], i0),
                                                            _mm256_castsi256_ps(_mm256_slli_epi32(i0, 28))));
      const __m256 w1 = _mm256_mul_ps(sv1, _mm256_blendv_ps(_mm256_permutevar8x32_ps(glo[t1], i1), _mm256_permutevar8x32_ps(ghi[t1], i1),
                                                            _mm256_castsi256_ps(_mm256_slli_epi32(i1, 28))));
      a0[g] = _mm256_fmadd_ps(w0, xv, a0[g]);
      a1[g] = _mm256_fmadd_ps(w1, xv, a1[g]);
    }
  }
}

PELIAS_TARGET_AVX2 void planes_seg_avx2(const std::uint8_t* o, int k_stored, int k_use, float* w) {
  const std::uint8_t* bits = o + 2 * k_stored;
  const __m256i bitsel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  const __m256 neg = _mm256_set1_ps(-0.0f);
  for (int pl = 0; pl < k_use; ++pl) {
    const __m256 s = _mm256_set1_ps(h2f(o + 2 * pl));
    const std::uint8_t* bp = bits + 32 * pl;
    for (int k = 0; k < kSeg / 8; ++k) {
      // bit clear: -s (sign bit set); bit set: +s. s x (+/-1) is exact, so this equals the reference's s * t.
      const __m256i set = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(bp[k]), bitsel), bitsel);
      const __m256 sv = _mm256_xor_ps(s, _mm256_andnot_ps(_mm256_castsi256_ps(set), neg));
      _mm256_storeu_ps(w + 8 * k, _mm256_add_ps(_mm256_loadu_ps(w + 8 * k), sv));
    }
  }
}

// One segment's products for nt tokens: each 8 weights loaded once for every token. Token tt's lanes (acc[tt][0..3] = lanes
// 0-31, value i to lane i mod 32) take the segment's fused multiply-adds in value order, as fma_seg_portable gives them, so
// the bits do not depend on nt or on which tokens share the call.
PELIAS_TARGET_AVX2 inline void fma_seg_nt_avx2(const float* w, const float* const* xs, std::int64_t off, int nt, __m256 (*acc)[4]) {
  for (int i = 0; i < kSeg; i += 32)
    for (int r = 0; r < 4; ++r) {
      const __m256 wv = _mm256_loadu_ps(w + i + 8 * r);
      for (int tt = 0; tt < nt; ++tt) acc[tt][r] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xs[tt] + off + i + 8 * r), acc[tt][r]);
    }
}

// A decoded row of k values (k a multiple of 32) against NT tokens, accumulators held in registers (NT fixed at compile
// time): each token's lanes take value i's product into lane i mod 32 in value order, exactly as fma_seg_nt_avx2 run
// segment after segment does, and the same tree reduces them.
PELIAS_TARGET_AVX2 float reduce_avx2(const __m256* acc);
template <int NT>
PELIAS_TARGET_AVX2 inline void dot_buf_nt_avx2(const float* w, const float* const* xs, std::int64_t k, float* out, std::int64_t ldo) {
  __m256 a[static_cast<std::size_t>(NT)][4];
  for (int tt = 0; tt < NT; ++tt)
    for (int r = 0; r < 4; ++r) a[tt][r] = _mm256_setzero_ps();
  for (std::int64_t i = 0; i < k; i += 32)
    for (int r = 0; r < 4; ++r) {
      const __m256 wv = _mm256_loadu_ps(w + i + 8 * r);
      for (int tt = 0; tt < NT; ++tt) a[tt][r] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xs[tt] + i + 8 * r), a[tt][r]);
    }
  for (int tt = 0; tt < NT; ++tt) out[tt * ldo] = reduce_avx2(a[tt]);
}

PELIAS_TARGET_AVX2 float reduce_avx2(const __m256* acc) {
  alignas(32) float l[32];
  for (int r = 0; r < 4; ++r) _mm256_store_ps(l + 8 * r, acc[r]);
  return reduce_portable(l);
}

// one 256-value segment through its AVX2 dequantizer (every value bit-identical to quant::dequantize_row's)
PELIAS_TARGET_AVX2 void deq_seg_avx2(q::Type t, const std::uint8_t* p, float* w) {
  switch (t) {
    case q::Type::q8_0: deq_q8_0(p, w); break;
    case q::Type::q4_k: deq_q4_k(p, w); break;
    case q::Type::q5_k: deq_q5_k(p, w); break;
    case q::Type::q6_k: deq_q6_k(p, w); break;
    case q::Type::iq4_xs: deq_iq4_xs(p, w); break;
    case q::Type::q4_tg: deq_q4_tg(p, w, 1); break;
    case q::Type::q4_tg2: deq_q4_tg(p, w, 2); break;
    case q::Type::q4_0: deq_q4_0(p, w); break;
    case q::Type::q4_1: deq_q4_1(p, w); break;
    case q::Type::q5_0: deq_q5_0(p, w); break;
    case q::Type::q5_1: deq_q5_1(p, w); break;
    case q::Type::iq4_nl: deq_iq4_nl(p, w); break;
    case q::Type::mxfp4: deq_mxfp4(p, w); break;
    case q::Type::q2_k: deq_q2_k(p, w); break;
    case q::Type::q3_k: deq_q3_k(p, w); break;
    case q::Type::q8_z: deq_q8_z(p, w); break;
    case q::Type::f16: deq_f16(p, w); break;
    case q::Type::bf16: deq_bf16(p, w); break;
    case q::Type::f32: std::memcpy(w, p, kSeg * sizeof(float)); break;
    case q::Type::iq2_xxs: deq_iq2_xxs(p, w); break;
    case q::Type::iq2_xs: deq_iq2_xs(p, w); break;
    case q::Type::iq2_s: deq_iq2_s(p, w); break;
    case q::Type::iq3_xxs: deq_iq3_xxs(p, w); break;
    case q::Type::iq3_s: deq_iq3_s(p, w); break;
    case q::Type::iq1_s: deq_iq1_s(p, w); break;
    case q::Type::iq1_m: deq_iq1_m(p, w); break;
    case q::Type::tq1_0: deq_tq1_0(p, w); break;
    case q::Type::tq2_0: deq_tq2_0(p, w); break;
    case q::Type::i2_s: deq_i2_s(p, w); break;
    case q::Type::nvfp4: deq_nvfp4(p, w); break;
    case q::Type::q7_z: deq_q7_z(p, w); break;
    case q::Type::q7_zt: deq_q7_zt(p, w); break;
    default: (void)q::dequantize_row(t, p, w, kSeg); break;
  }
}

PELIAS_TARGET_AVX2 void dot_row_avx2(q::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* pl, const float* const* xs,
                                     int nt, float* out) {
  alignas(32) float w[kSeg];
  __m256 acc[4][4];
  for (int tt = 0; tt < nt; ++tt)
    for (int r = 0; r < 4; ++r) acc[tt][r] = _mm256_setzero_ps();
  const std::uint64_t seg_bytes = q::row_bytes(static_cast<std::uint32_t>(t), kSeg);
  const std::int64_t nseg = k / kSeg;
  const std::uint8_t* pd = pl ? pl->data : nullptr;
  const std::int64_t pbytes = pl ? 34 * static_cast<std::int64_t>(pl->k_stored) : 0;
  const bool tg = t == q::Type::q4_tg || t == q::Type::q4_tg2;
  for (std::int64_t sgi = 0; sgi < nseg; ++sgi) {
    const std::uint8_t* p = row + static_cast<std::uint64_t>(sgi) * seg_bytes;
    const bool seg_planes = pd && (!pl->mask || ((pl->mask[sgi / 8] >> (sgi % 8)) & 1));
    if (tg && !seg_planes) {
      const float* xo[4];
      for (int tt = 0; tt < nt; ++tt) xo[tt] = xs[tt] + sgi * kSeg;
      fused_q4_tg_seg(p, t == q::Type::q4_tg2 ? 2 : 1, xo, nt, acc);
      continue;
    }
    deq_seg_avx2(t, p, w);
    if (pd) {
      if (!pl->mask) {
        planes_seg_avx2(pd + sgi * pbytes, pl->k_stored, pl->k_use, w);
      } else if ((pl->mask[sgi / 8] >> (sgi % 8)) & 1) {
        planes_seg_avx2(pd, pl->k_stored, pl->k_use, w);
        pd += pbytes;
      }
    }
    fma_seg_nt_avx2(w, xs, sgi * kSeg, nt, acc);
  }
  for (int tt = 0; tt < nt; ++tt) out[tt] = reduce_avx2(acc[tt]);
}


// ---- the int8-activation path

bool avx_ints(q::Type t) {
  switch (t) {
    case q::Type::q8_0: case q::Type::q8_z: case q::Type::q4_0: case q::Type::q4_1: case q::Type::iq4_nl: case q::Type::mxfp4:
    case q::Type::iq4_xs: case q::Type::q4_k: case q::Type::q5_k: case q::Type::q6_k:
      return true;
    default:
      return false;
  }
}

PELIAS_TARGET_AVX2 inline void st32(std::int8_t* y, int at, __m256i v) { _mm256_store_si256(reinterpret_cast<__m256i*>(y + at), v); }

PELIAS_TARGET_AVX2 void ints_avx2(q::Type t, const std::uint8_t* p, std::int8_t* y) {
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  switch (t) {
    case q::Type::q8_0:
      for (int b = 0; b < 8; ++b) st32(y, 32 * b, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 34 * b + 2)));
      return;
    case q::Type::q8_z:
      for (int j = 0; j < 8; ++j) st32(y, 32 * j, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8 + 32 * j)));
      return;
    case q::Type::q4_0:
      for (int b = 0; b < 8; ++b) st32(y, 32 * b, _mm256_sub_epi8(nibbles32(p + 18 * b + 2), _mm256_set1_epi8(8)));
      return;
    case q::Type::q4_1:
      for (int b = 0; b < 8; ++b) st32(y, 32 * b, nibbles32(p + 20 * b + 4));
      return;
    case q::Type::iq4_nl: case q::Type::mxfp4: {
      const __m256i tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(t == q::Type::mxfp4 ? kFp4 : kIq4)));
      const int bytes = t == q::Type::mxfp4 ? 17 : 18, qo = t == q::Type::mxfp4 ? 1 : 2;
      for (int b = 0; b < 8; ++b) st32(y, 32 * b, _mm256_shuffle_epi8(tab, nibbles32(p + bytes * b + qo)));
      return;
    }
    case q::Type::iq4_xs: {
      const __m256i tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kIq4)));
      for (int ib = 0; ib < 8; ++ib) st32(y, 32 * ib, _mm256_shuffle_epi8(tab, nibbles32(p + 8 + 16 * ib)));
      return;
    }
    case q::Type::q4_k: case q::Type::q5_k: {
      const bool five = t == q::Type::q5_k;
      const std::uint8_t* qs = p + (five ? 48 : 16);
      const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 16)), one = _mm256_set1_epi8(1);
      for (int j = 0; j < 4; ++j, qs += 32) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
        __m256i lo = _mm256_and_si256(v, m4), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
        if (five) {
          lo = _mm256_add_epi8(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * j), one), 4));
          hi = _mm256_add_epi8(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * j + 1), one), 4));
        }
        st32(y, 64 * j, lo);
        st32(y, 64 * j + 32, hi);
      }
      return;
    }
    case q::Type::q6_k: {
      const __m256i m3 = _mm256_set1_epi8(3), o32 = _mm256_set1_epi8(32);
      for (int h = 0; h < 2; ++h) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 64 * h));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 64 * h + 32));
        const __m256i hb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 128 + 32 * h));
        st32(y, 128 * h + 0, _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(a, m4), q6_hi2(hb, 0, m3)), o32));
        st32(y, 128 * h + 32, _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(b, m4), q6_hi2(hb, 2, m3)), o32));
        st32(y, 128 * h + 64, _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a, 4), m4), q6_hi2(hb, 4, m3)), o32));
        st32(y, 128 * h + 96, _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b, 4), m4), q6_hi2(hb, 6, m3)), o32));
      }
      return;
    }
    default:
      return;
  }
}

PELIAS_TARGET_AVX2 void dot_row_q8_avx2(q::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* pl, const ActQ8x32* const* acts,
                                        int nt, float* out) {
  Seg8 o;
  __m256 acc[4];
  float accp[4] = {};
  const bool mins = has_mins(t);
  for (int tt = 0; tt < nt; ++tt) acc[tt] = _mm256_setzero_ps();
  const std::uint64_t seg_bytes = q::row_bytes(static_cast<std::uint32_t>(t), kSeg);
  const std::uint8_t* pd = pl ? pl->data : nullptr;
  const std::int64_t pbytes = pl ? 34 * static_cast<std::int64_t>(pl->k_stored) : 0;
  const bool vi = avx_ints(t);
  const __m256i ones = _mm256_set1_epi16(1);
  for (std::int64_t sgi = 0; sgi < k / kSeg; ++sgi) {
    const std::uint8_t* p = row + static_cast<std::uint64_t>(sgi) * seg_bytes;
    seg8_portable(t, p, o, !vi);
    if (vi) ints_avx2(t, p, o.q);
    if (pd) {
      const std::uint8_t* po = nullptr;
      if (!pl->mask) po = pd + sgi * pbytes;
      else if ((pl->mask[sgi / 8] >> (sgi % 8)) & 1) {
        po = pd;
        pd += pbytes;
      }
      if (po)
        for (int tt = 0; tt < nt; ++tt) accp[tt] += planes_q8_seg(po, pl->k_stored, pl->k_use, *acts[tt], sgi);
    }
    for (int b = 0; b < 8; ++b) {
      const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(o.q + 32 * b));
      const __m256i aw = _mm256_abs_epi8(w);
      for (int tt = 0; tt < nt; ++tt) {
        const ActQ8x32& a = *acts[tt];
        const std::size_t i0 = static_cast<std::size_t>(sgi * kSeg + 32 * b);
        const __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q.data() + i0));
        const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(qa, w)), ones);
        const float sb = a.s[i0 / 32];
        const __m256 scl = _mm256_set_m128(_mm_set1_ps(o.sw[2 * b + 1] * sb), _mm_set1_ps(o.sw[2 * b] * sb));
        acc[tt] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), scl, acc[tt]);
        if (mins) {  // the mins enter lanes 0 and 4 block by block, so a large activation offset cancels as it goes
          const float m0 = -(sb * (o.mw[2 * b] * static_cast<float>(a.sum16[i0 / 16])));
          const float m1 = -(sb * (o.mw[2 * b + 1] * static_cast<float>(a.sum16[i0 / 16 + 1])));
          acc[tt] = _mm256_add_ps(acc[tt], _mm256_setr_ps(m0, 0.0f, 0.0f, 0.0f, m1, 0.0f, 0.0f, 0.0f));
        }
      }
    }
  }
  for (int tt = 0; tt < nt; ++tt) {
    alignas(32) float l[8];
    _mm256_store_ps(l, acc[tt]);
    out[tt] = reduce8(l) + accp[tt];
  }
}

// ---- many tokens per block of rows (GP3-10 batch fix, 2026-10-30): each row of the block decoded once, then each group of
// four tokens run across every row of the block (its inputs stay in cache across the rows, each row's weights across the
// four tokens); every (row, token) value summed exactly as dot_row / dot_row_q8 sums it, so the same bits

// row r's planes for segment sgi (null: none), walking the block-selective storage in order
struct PlaneWalk {
  const RowPlanes* pl;
  const std::uint8_t* pd;
  std::int64_t pbytes;
  explicit PlaneWalk(const RowPlanes* p) : pl(p), pd(p ? p->data : nullptr), pbytes(p ? 34 * static_cast<std::int64_t>(p->k_stored) : 0) {}
  const std::uint8_t* next(std::int64_t sgi) {
    if (!pd) return nullptr;
    if (!pl->mask) return pd + sgi * pbytes;
    if ((pl->mask[sgi / 8] >> (sgi % 8)) & 1) {
      const std::uint8_t* o = pd;
      pd += pbytes;
      return o;
    }
    return nullptr;
  }
};

PELIAS_TARGET_AVX2 void dot_rows_n_avx2(q::Type t, const std::uint8_t* const* rows, const RowPlanes* const* pls, int nr, std::int64_t k,
                                        const float* const* xs, std::int64_t n, float* out, std::int64_t ldo) {
  thread_local std::vector<float> wbuf;
  const std::size_t need = static_cast<std::size_t>(nr) * static_cast<std::size_t>(k);
  if (wbuf.size() < need) wbuf.resize(need);
  const std::uint64_t seg_bytes = q::row_bytes(static_cast<std::uint32_t>(t), kSeg);
  const std::int64_t nseg = k / kSeg;
  for (int r = 0; r < nr; ++r) {
    float* w = wbuf.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(k);
    PlaneWalk pw(pls ? pls[r] : nullptr);
    for (std::int64_t sgi = 0; sgi < nseg; ++sgi) {
      float* ws = w + sgi * kSeg;
      deq_seg_avx2(t, rows[r] + static_cast<std::uint64_t>(sgi) * seg_bytes, ws);
      if (const std::uint8_t* po = pw.next(sgi)) planes_seg_avx2(po, pw.pl->k_stored, pw.pl->k_use, ws);
    }
  }
  for (std::int64_t t0 = 0; t0 < n; t0 += 4) {
    const int nt = static_cast<int>(std::min<std::int64_t>(4, n - t0));
    for (int r = 0; r < nr; ++r) {
      const float* w = wbuf.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(k);
      float* o = out + t0 * ldo + r;
      switch (nt) {
        case 4: dot_buf_nt_avx2<4>(w, xs + t0, k, o, ldo); break;
        case 3: dot_buf_nt_avx2<3>(w, xs + t0, k, o, ldo); break;
        case 2: dot_buf_nt_avx2<2>(w, xs + t0, k, o, ldo); break;
        default: dot_buf_nt_avx2<1>(w, xs + t0, k, o, ldo); break;
      }
    }
  }
}

PELIAS_TARGET_AVX2 void dot_rows_q8_n_avx2(q::Type t, const std::uint8_t* const* rows, const RowPlanes* const* pls, int nr, std::int64_t k,
                                           const ActQ8x32* acts, std::int64_t n, float* out, std::int64_t ldo) {
  thread_local std::vector<Seg8> segs;
  thread_local std::vector<const std::uint8_t*> po;
  const std::int64_t nseg = k / kSeg;
  const std::size_t need = static_cast<std::size_t>(nr) * static_cast<std::size_t>(nseg);
  if (segs.size() < need) segs.resize(need);
  if (po.size() < need) po.resize(need);
  const bool mins = has_mins(t);
  const std::uint64_t seg_bytes = q::row_bytes(static_cast<std::uint32_t>(t), kSeg);
  const bool vi = avx_ints(t);
  for (int r = 0; r < nr; ++r) {
    PlaneWalk pw(pls ? pls[r] : nullptr);
    for (std::int64_t sgi = 0; sgi < nseg; ++sgi) {
      const std::uint8_t* p = rows[r] + static_cast<std::uint64_t>(sgi) * seg_bytes;
      const std::size_t at = static_cast<std::size_t>(r) * static_cast<std::size_t>(nseg) + static_cast<std::size_t>(sgi);
      seg8_portable(t, p, segs[at], !vi);
      if (vi) ints_avx2(t, p, segs[at].q);
      po[at] = pw.next(sgi);
    }
  }
  const __m256i ones = _mm256_set1_epi16(1);
  for (std::int64_t t0 = 0; t0 < n; t0 += 4) {
    const int nt = static_cast<int>(std::min<std::int64_t>(4, n - t0));
    for (int r = 0; r < nr; ++r) {
      const RowPlanes* pl = pls ? pls[r] : nullptr;
      __m256 acc[4];
      float accp[4] = {};
      for (int tt = 0; tt < nt; ++tt) acc[tt] = _mm256_setzero_ps();
      for (std::int64_t sgi = 0; sgi < nseg; ++sgi) {
        const std::size_t at = static_cast<std::size_t>(r) * static_cast<std::size_t>(nseg) + static_cast<std::size_t>(sgi);
        const Seg8& o = segs[at];
        if (const std::uint8_t* pp = po[at])
          for (int tt = 0; tt < nt; ++tt) accp[tt] += planes_q8_seg(pp, pl->k_stored, pl->k_use, acts[t0 + tt], sgi);
        for (int b = 0; b < 8; ++b) {
          const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(o.q + 32 * b));
          const __m256i aw = _mm256_abs_epi8(w);
          for (int tt = 0; tt < nt; ++tt) {
            const ActQ8x32& a = acts[t0 + tt];
            const std::size_t i0 = static_cast<std::size_t>(sgi * kSeg + 32 * b);
            const __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q.data() + i0));
            const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(qa, w)), ones);
            const float sb = a.s[i0 / 32];
            const __m256 scl = _mm256_set_m128(_mm_set1_ps(o.sw[2 * b + 1] * sb), _mm_set1_ps(o.sw[2 * b] * sb));
            acc[tt] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), scl, acc[tt]);
            if (mins) {
              const float m0 = -(sb * (o.mw[2 * b] * static_cast<float>(a.sum16[i0 / 16])));
              const float m1 = -(sb * (o.mw[2 * b + 1] * static_cast<float>(a.sum16[i0 / 16 + 1])));
              acc[tt] = _mm256_add_ps(acc[tt], _mm256_setr_ps(m0, 0.0f, 0.0f, 0.0f, m1, 0.0f, 0.0f, 0.0f));
            }
          }
        }
      }
      for (int tt = 0; tt < nt; ++tt) {
        alignas(32) float l[8];
        _mm256_store_ps(l, acc[tt]);
        out[(t0 + tt) * ldo + r] = reduce8(l) + accp[tt];
      }
    }
  }
}

#endif  // PELIAS_KERN_X86

void dot_row_portable(q::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* pl, const float* const* xs, int nt,
                      float* out) {
  float w[kSeg];
  float acc[4][kLanes] = {};
  const std::uint64_t seg_bytes = q::row_bytes(static_cast<std::uint32_t>(t), kSeg);
  const std::int64_t nseg = k / kSeg;
  const std::uint8_t* pd = pl ? pl->data : nullptr;
  const std::int64_t pbytes = pl ? 34 * static_cast<std::int64_t>(pl->k_stored) : 0;
  for (std::int64_t sgi = 0; sgi < nseg; ++sgi) {
    (void)q::dequantize_row(t, row + static_cast<std::uint64_t>(sgi) * seg_bytes, w, kSeg);
    if (pd) {
      if (!pl->mask) {
        planes_seg_portable(pd + sgi * pbytes, pl->k_stored, pl->k_use, w);
      } else if ((pl->mask[sgi / 8] >> (sgi % 8)) & 1) {
        planes_seg_portable(pd, pl->k_stored, pl->k_use, w);
        pd += pbytes;
      }
    }
    for (int tt = 0; tt < nt; ++tt) fma_seg_portable(w, xs[tt] + sgi * kSeg, acc[tt]);
  }
  for (int tt = 0; tt < nt; ++tt) out[tt] = reduce_portable(acc[tt]);
}

// a row not a whole number of segments: dequantized whole, planes added as the reference adds them, the same lanes
void dot_row_whole(q::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* pl, const float* const* xs, int nt, float* out) {
  std::vector<float> w(static_cast<std::size_t>(k));
  (void)q::dequantize_row(t, row, w.data(), k);
  if (pl && pl->data) {
    if (pl->mask) q::plane_add_row_sparse(pl->data, pl->mask, k, pl->k_stored, pl->k_use, w.data());
    else q::plane_add_row(pl->data, k, pl->k_stored, pl->k_use, w.data());
  }
  for (int tt = 0; tt < nt; ++tt) {
    float acc[kLanes] = {};
    for (std::int64_t i = 0; i < k; ++i) acc[i % kLanes] = std::fma(w[static_cast<std::size_t>(i)], xs[tt][i], acc[i % kLanes]);
    out[tt] = reduce_portable(acc);
  }
}

void dot_row_q8_portable(q::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* pl, const ActQ8x32* const* acts, int nt,
                         float* out) {
  Seg8 o;
  float acc[4][8] = {}, accp[4] = {};
  const bool mins = has_mins(t);
  const std::uint64_t seg_bytes = q::row_bytes(static_cast<std::uint32_t>(t), kSeg);
  const std::uint8_t* pd = pl ? pl->data : nullptr;
  const std::int64_t pbytes = pl ? 34 * static_cast<std::int64_t>(pl->k_stored) : 0;
  for (std::int64_t sgi = 0; sgi < k / kSeg; ++sgi) {
    const std::uint8_t* p = row + static_cast<std::uint64_t>(sgi) * seg_bytes;
    seg8_portable(t, p, o);
    if (pd) {
      const std::uint8_t* po = nullptr;
      if (!pl->mask) po = pd + sgi * pbytes;
      else if ((pl->mask[sgi / 8] >> (sgi % 8)) & 1) {
        po = pd;
        pd += pbytes;
      }
      if (po)
        for (int tt = 0; tt < nt; ++tt) accp[tt] += planes_q8_seg(po, pl->k_stored, pl->k_use, *acts[tt], sgi);
    }
    for (int b = 0; b < 8; ++b)
      for (int tt = 0; tt < nt; ++tt) {
        const ActQ8x32& a = *acts[tt];
        const std::size_t i0 = static_cast<std::size_t>(sgi * kSeg + 32 * b);
        std::int32_t lane[8];
        isum_lanes_portable(o.q + 32 * b, a.q.data() + i0, lane);
        const float sb = a.s[i0 / 32];
        const float s0 = o.sw[2 * b] * sb, s1 = o.sw[2 * b + 1] * sb;
        for (int j = 0; j < 8; ++j) acc[tt][j] = std::fma(static_cast<float>(lane[j]), j < 4 ? s0 : s1, acc[tt][j]);
        if (mins) {
          const float m0 = -(sb * (o.mw[2 * b] * static_cast<float>(a.sum16[i0 / 16])));
          const float m1 = -(sb * (o.mw[2 * b + 1] * static_cast<float>(a.sum16[i0 / 16 + 1])));
          const float add[8] = {m0, 0.0f, 0.0f, 0.0f, m1, 0.0f, 0.0f, 0.0f};
          for (int j = 0; j < 8; ++j) acc[tt][j] = acc[tt][j] + add[j];
        }
      }
  }
  for (int tt = 0; tt < nt; ++tt) out[tt] = reduce8(acc[tt]) + accp[tt];
}

void act_fill(ActQ8x32& a, std::int64_t k) {
  a.k = k;
  a.q.resize(static_cast<std::size_t>(k));
  a.s.resize(static_cast<std::size_t>(k / 32));
  a.sum16.resize(static_cast<std::size_t>(k / 16));
  a.x.resize(static_cast<std::size_t>(k));
}

void act_block(const float* x, ActQ8x32& a, std::int64_t b, float scale) {  // q = clamp(nearbyint(x * (1 / scale)))
  const float inv = 1.0f / scale;
  a.s[static_cast<std::size_t>(b)] = scale;
  std::int32_t s0 = 0, s1 = 0;
  for (int j = 0; j < 32; ++j) {
    const float qf = std::min(127.0f, std::max(-127.0f, std::nearbyint(x[j] * inv)));
    const auto qi = static_cast<std::int8_t>(qf);
    a.q[static_cast<std::size_t>(32 * b + j)] = qi;
    (j < 16 ? s0 : s1) += qi;
  }
  a.sum16[static_cast<std::size_t>(2 * b)] = s0;
  a.sum16[static_cast<std::size_t>(2 * b + 1)] = s1;
}

}  // namespace

#if defined(PELIAS_KERN_X86)
namespace {
// v45 (Fable review: the CPU's input rounding is scalar nearbyint, 0.33 ms per token on the critical path): the same block in
// AVX2. amax by max (order-free), the same scalar scale and 1 / scale, the same IEEE multiply per value, round to nearest even
// (= nearbyint in the default rounding mode), the same clamp to +/-127, the same integer sums, x = scale * q: bit for bit.
PELIAS_TARGET_AVX2 void act_q8x32_quantize_avx2(const float* x, std::int64_t k, ActQ8x32& a) {
  const __m256 sign = _mm256_set1_ps(-0.0f), lo = _mm256_set1_ps(-127.0f), hi = _mm256_set1_ps(127.0f);
  for (std::int64_t b = 0; b < k / 32; ++b) {
    const float* xb = x + 32 * b;
    __m256 v[4], m = _mm256_setzero_ps();
    for (int r = 0; r < 4; ++r) {
      v[r] = _mm256_loadu_ps(xb + 8 * r);
      m = _mm256_max_ps(m, _mm256_andnot_ps(sign, v[r]));
    }
    alignas(32) float mm[8];
    _mm256_store_ps(mm, m);
    float amax = 0.0f;
    for (int j = 0; j < 8; ++j) amax = std::max(amax, mm[j]);
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    a.s[static_cast<std::size_t>(b)] = scale;
    const __m256 iv = _mm256_set1_ps(inv), sv = _mm256_set1_ps(scale);
    alignas(32) std::int32_t qi[32];
    for (int r = 0; r < 4; ++r) {
      const __m256 q = _mm256_min_ps(hi, _mm256_max_ps(lo, _mm256_round_ps(_mm256_mul_ps(v[r], iv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
      const __m256i qn = _mm256_cvttps_epi32(q);
      _mm256_store_si256(reinterpret_cast<__m256i*>(qi + 8 * r), qn);
      // from the integer, as the scalar form does (a value rounded to -0 gives +0 here, as there: bit for bit)
      _mm256_storeu_ps(a.x.data() + 32 * b + 8 * r, _mm256_mul_ps(sv, _mm256_cvtepi32_ps(qn)));
    }
    std::int32_t s0 = 0, s1 = 0;
    for (int j = 0; j < 32; ++j) {
      a.q[static_cast<std::size_t>(32 * b + j)] = static_cast<std::int8_t>(qi[j]);
      (j < 16 ? s0 : s1) += qi[j];
    }
    a.sum16[static_cast<std::size_t>(2 * b)] = s0;
    a.sum16[static_cast<std::size_t>(2 * b + 1)] = s1;
  }
}
}  // namespace
#endif

void act_q8x32_quantize(const float* x, std::int64_t k, ActQ8x32& a) {
#if defined(PELIAS_KERN_X86)
  if (active_isa() == Isa::avx2) {
    act_fill(a, k);
    act_q8x32_quantize_avx2(x, k, a);
    return;
  }
#endif
  act_q8x32_quantize_ref(x, k, a);
}

void act_q8x32_quantize_ref(const float* x, std::int64_t k, ActQ8x32& a) {
  act_fill(a, k);
  for (std::int64_t b = 0; b < k / 32; ++b) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32 * b + j]));
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    act_block(x + 32 * b, a, b, scale);
    for (int j = 0; j < 32; ++j)
      a.x[static_cast<std::size_t>(32 * b + j)] = scale * static_cast<float>(a.q[static_cast<std::size_t>(32 * b + j)]);
  }
}

void act_q8x32_from_rounded(const float* xr, std::int64_t k, ActQ8x32& a) {
  act_fill(a, k);
  std::copy(xr, xr + k, a.x.begin());
  for (std::int64_t b = 0; b < k / 32; ++b) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xr[32 * b + j]));
    act_block(xr + 32 * b, a, b, amax > 0.0f ? amax / 127.0f : 1.0f);
  }
}

bool dot_q8_has_int_path(q::Type t) { return int_path(t); }

void dot_row_q8(q::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* planes, const ActQ8x32* const* acts, int nt, float* out) {
  if (planes && !planes->data) planes = nullptr;
  if (planes && planes->k_use > planes->k_stored) {
    RowPlanes c = *planes;
    c.k_use = c.k_stored;
    dot_row_q8(t, row, k, &c, acts, nt, out);
    return;
  }
  if (!int_path(t) || k % kSeg != 0) {  // the float path on the rounded values
    const float* xs[4] = {};
    for (int tt = 0; tt < nt; ++tt) xs[tt] = acts[tt]->x.data();
    dot_row(t, row, k, planes, xs, nt, out);
    return;
  }
#if defined(PELIAS_KERN_X86)
  if (active_isa() == Isa::avx2) {
    dot_row_q8_avx2(t, row, k, planes, acts, nt, out);
    return;
  }
#endif
  dot_row_q8_portable(t, row, k, planes, acts, nt, out);
}

namespace {
// a planes pointer as dot_row reads it: none without data, k_use capped at k_stored
const RowPlanes* clamp_planes(const RowPlanes* p, RowPlanes& c) {
  if (!p || !p->data) return nullptr;
  if (p->k_use <= p->k_stored) return p;
  c = *p;
  c.k_use = c.k_stored;
  return &c;
}
constexpr int kMaxBlockRows = 64;
}  // namespace

void dot_rows_n(q::Type t, const std::uint8_t* const* rows, const RowPlanes* const* planes, int nr, std::int64_t k, const float* const* xs,
                std::int64_t n, float* out, std::int64_t ldo) {
  if (nr <= 0) return;
  if (nr > kMaxBlockRows) {
    for (int r0 = 0; r0 < nr; r0 += kMaxBlockRows)
      dot_rows_n(t, rows + r0, planes ? planes + r0 : nullptr, std::min(kMaxBlockRows, nr - r0), k, xs, n, out + r0, ldo);
    return;
  }
  RowPlanes cl[kMaxBlockRows];
  const RowPlanes* pp[kMaxBlockRows];
  for (int r = 0; r < nr; ++r) pp[r] = clamp_planes(planes ? planes[r] : nullptr, cl[r]);
#if defined(PELIAS_KERN_X86)
  if (active_isa() == Isa::avx2 && k % kSeg == 0 && q::row_bytes(static_cast<std::uint32_t>(t), kSeg) != 0) {
    dot_rows_n_avx2(t, rows, pp, nr, k, xs, n, out, ldo);
    return;
  }
#endif
  for (int r = 0; r < nr; ++r)
    for (std::int64_t t0 = 0; t0 < n; t0 += 4) {
      const int nt = static_cast<int>(std::min<std::int64_t>(4, n - t0));
      float o[4];
      dot_row(t, rows[r], k, pp[r], xs + t0, nt, o);
      for (int tt = 0; tt < nt; ++tt) out[(t0 + tt) * ldo + r] = o[tt];
    }
}

void dot_rows_q8_n(q::Type t, const std::uint8_t* const* rows, const RowPlanes* const* planes, int nr, std::int64_t k, const ActQ8x32* acts,
                   std::int64_t n, float* out, std::int64_t ldo) {
  if (nr <= 0) return;
  if (!int_path(t) || k % kSeg != 0) {  // the float path on the rounded values, as dot_row_q8 takes it
    thread_local std::vector<const float*> xs;
    xs.resize(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) xs[static_cast<std::size_t>(i)] = acts[i].x.data();
    dot_rows_n(t, rows, planes, nr, k, xs.data(), n, out, ldo);
    return;
  }
  if (nr > kMaxBlockRows) {
    for (int r0 = 0; r0 < nr; r0 += kMaxBlockRows)
      dot_rows_q8_n(t, rows + r0, planes ? planes + r0 : nullptr, std::min(kMaxBlockRows, nr - r0), k, acts, n, out + r0, ldo);
    return;
  }
  RowPlanes cl[kMaxBlockRows];
  const RowPlanes* pp[kMaxBlockRows];
  for (int r = 0; r < nr; ++r) pp[r] = clamp_planes(planes ? planes[r] : nullptr, cl[r]);
#if defined(PELIAS_KERN_X86)
  if (active_isa() == Isa::avx2) {
    dot_rows_q8_n_avx2(t, rows, pp, nr, k, acts, n, out, ldo);
    return;
  }
#endif
  for (int r = 0; r < nr; ++r)
    for (std::int64_t t0 = 0; t0 < n; t0 += 4) {
      const int nt = static_cast<int>(std::min<std::int64_t>(4, n - t0));
      const ActQ8x32* as[4];
      for (int tt = 0; tt < nt; ++tt) as[tt] = acts + t0 + tt;
      float o[4];
      dot_row_q8(t, rows[r], k, pp[r], as, nt, o);
      for (int tt = 0; tt < nt; ++tt) out[(t0 + tt) * ldo + r] = o[tt];
    }
}

void dot_row_n(q::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* planes, const float* const* xs, std::int64_t n, float* out,
               std::int64_t ldo) {
  dot_rows_n(t, &row, planes ? &planes : nullptr, 1, k, xs, n, out, ldo);
}

void dot_row_q8_n(q::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* planes, const ActQ8x32* acts, std::int64_t n, float* out,
                  std::int64_t ldo) {
  dot_rows_q8_n(t, &row, planes ? &planes : nullptr, 1, k, acts, n, out, ldo);
}

namespace {
}  // namespace

Isa best_isa() {
  static const bool avx2 = cpu_has_avx2_fma();
  return avx2 ? Isa::avx2 : Isa::portable;
}

Isa active_isa() {
  const int f = g_forced.load(std::memory_order_relaxed);
  if (f == static_cast<int>(Isa::portable)) return Isa::portable;
  return best_isa();
}

void force_isa(Isa isa) { g_forced.store(static_cast<int>(isa), std::memory_order_relaxed); }

bool dot_supported(q::Type t) { return q::can_dequantize(t); }
bool dot_has_fast_path(q::Type t) { return is_fast(t); }

#if defined(PELIAS_KERN_X86)
namespace {
PELIAS_TARGET_AVX2 void dot_rows2_tg_avx2(int version, const std::uint8_t* r0, const std::uint8_t* r1, std::int64_t k, std::uint64_t seg_bytes,
                                          const float* x, float* out) {
  __m256 glo[4], ghi[4];
  for (int t = 0; t < 4; ++t) {
    const float* g = tg_grid(t, version);
    glo[t] = _mm256_loadu_ps(g);
    ghi[t] = _mm256_loadu_ps(g + 8);
  }
  __m256 a0[4], a1[4];
  for (int r = 0; r < 4; ++r) {
    a0[r] = _mm256_setzero_ps();
    a1[r] = _mm256_setzero_ps();
  }
  for (std::int64_t sgi = 0; sgi < k / kSeg; ++sgi)
    fused_q4_tg_seg2(r0 + static_cast<std::uint64_t>(sgi) * seg_bytes, r1 + static_cast<std::uint64_t>(sgi) * seg_bytes, glo, ghi, x + sgi * kSeg, a0, a1);
  out[0] = reduce_avx2(a0);
  out[1] = reduce_avx2(a1);
}
// v45b (author 17:19-17:21): a tile of row pairs in one call, the grids loaded once per tile. Pair i is the rows at
// b0 + i * stride and b1 + i * stride; each pair's arithmetic is dot_rows2_tg_avx2's (bit for bit).
PELIAS_TARGET_AVX2 void dot_tile2_tg_avx2(int version, const std::uint8_t* b0, const std::uint8_t* b1, std::uint64_t stride, std::int64_t n, std::int64_t k,
                                          std::uint64_t seg_bytes, const float* x, float* o0, float* o1, std::int64_t os) {
  __m256 glo[4], ghi[4];
  for (int t = 0; t < 4; ++t) {
    const float* g = tg_grid(t, version);
    glo[t] = _mm256_loadu_ps(g);
    ghi[t] = _mm256_loadu_ps(g + 8);
  }
  const std::int64_t nseg = k / kSeg;
  for (std::int64_t i = 0; i < n; ++i) {
    const std::uint8_t* r0 = b0 + static_cast<std::uint64_t>(i) * stride;
    const std::uint8_t* r1 = b1 + static_cast<std::uint64_t>(i) * stride;
    __m256 a0[4], a1[4];
    for (int r = 0; r < 4; ++r) {
      a0[r] = _mm256_setzero_ps();
      a1[r] = _mm256_setzero_ps();
    }
    for (std::int64_t sgi = 0; sgi < nseg; ++sgi)
      fused_q4_tg_seg2(r0 + static_cast<std::uint64_t>(sgi) * seg_bytes, r1 + static_cast<std::uint64_t>(sgi) * seg_bytes, glo, ghi, x + sgi * kSeg, a0, a1);
    o0[i * os] = reduce_avx2(a0);
    o1[i * os] = reduce_avx2(a1);
  }
}
}  // namespace
#endif

void dot_rows2_f32(q::Type t, const std::uint8_t* r0, const std::uint8_t* r1, std::int64_t k, const float* x, float* out) {
#if defined(PELIAS_KERN_X86)
  if (active_isa() == Isa::avx2 && (t == q::Type::q4_tg || t == q::Type::q4_tg2) && k % kSeg == 0) {
    dot_rows2_tg_avx2(t == q::Type::q4_tg2 ? 2 : 1, r0, r1, k, q::row_bytes(static_cast<std::uint32_t>(t), kSeg), x, out);
    return;
  }
#endif
  const float* xs[1] = {x};
  dot_row(t, r0, k, nullptr, xs, 1, out);
  dot_row(t, r1, k, nullptr, xs, 1, out + 1);
}

void dot_tile2_f32(q::Type t, const std::uint8_t* b0, const std::uint8_t* b1, std::uint64_t stride, std::int64_t n, std::int64_t k, const float* x, float* o0,
                   float* o1, std::int64_t os) {
#if defined(PELIAS_KERN_X86)
  if (active_isa() == Isa::avx2 && (t == q::Type::q4_tg || t == q::Type::q4_tg2) && k % kSeg == 0) {
    dot_tile2_tg_avx2(t == q::Type::q4_tg2 ? 2 : 1, b0, b1, stride, n, k, q::row_bytes(static_cast<std::uint32_t>(t), kSeg), x, o0, o1, os);
    return;
  }
#endif
  for (std::int64_t i = 0; i < n; ++i) {
    float o2[2];
    dot_rows2_f32(t, b0 + static_cast<std::uint64_t>(i) * stride, b1 + static_cast<std::uint64_t>(i) * stride, k, x, o2);
    o0[i * os] = o2[0];
    o1[i * os] = o2[1];
  }
}

void dot_row(q::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* planes, const float* const* xs, int nt, float* out) {
  if (planes && !planes->data) planes = nullptr;
  if (planes && planes->k_use > planes->k_stored) {
    RowPlanes c = *planes;
    c.k_use = c.k_stored;
    dot_row(t, row, k, &c, xs, nt, out);
    return;
  }
  if (k % kSeg != 0 || q::row_bytes(static_cast<std::uint32_t>(t), kSeg) == 0) {
    dot_row_whole(t, row, k, planes, xs, nt, out);
    return;
  }
#if defined(PELIAS_KERN_X86)
  if (active_isa() == Isa::avx2) {
    dot_row_avx2(t, row, k, planes, xs, nt, out);
    return;
  }
#endif
  dot_row_portable(t, row, k, planes, xs, nt, out);
}

}  // namespace pelias::kern

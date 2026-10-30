// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// Fast CPU row dot products (kern/cpu_dot.h). The dequantizers below compute every value with the same float expression,
// in the same order, as quant/quant_cpu.cpp's dequantize_* (read there, not copied from elsewhere); the summation is ours.
#include "kern/cpu_dot.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

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
    case q::Type::q8_z: case q::Type::f16: case q::Type::bf16: case q::Type::f32:
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

std::uint32_t u32le(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) | (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
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

PELIAS_TARGET_AVX2 inline void fma_seg_avx2(const float* w, const float* x, __m256* acc) {  // acc[4] = lanes 0-31
  for (int i = 0; i < kSeg; i += 32) {
    acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), acc[0]);
    acc[1] = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), acc[1]);
    acc[2] = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 16), _mm256_loadu_ps(x + i + 16), acc[2]);
    acc[3] = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 24), _mm256_loadu_ps(x + i + 24), acc[3]);
  }
}

PELIAS_TARGET_AVX2 float reduce_avx2(const __m256* acc) {
  alignas(32) float l[32];
  for (int r = 0; r < 4; ++r) _mm256_store_ps(l + 8 * r, acc[r]);
  return reduce_portable(l);
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
  for (std::int64_t sgi = 0; sgi < nseg; ++sgi) {
    const std::uint8_t* p = row + static_cast<std::uint64_t>(sgi) * seg_bytes;
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
      case q::Type::f32: std::memcpy(w, p, sizeof w); break;
      default: (void)q::dequantize_row(t, p, w, kSeg); break;
    }
    if (pd) {
      if (!pl->mask) {
        planes_seg_avx2(pd + sgi * pbytes, pl->k_stored, pl->k_use, w);
      } else if ((pl->mask[sgi / 8] >> (sgi % 8)) & 1) {
        planes_seg_avx2(pd, pl->k_stored, pl->k_use, w);
        pd += pbytes;
      }
    }
    for (int tt = 0; tt < nt; ++tt) fma_seg_avx2(w, xs[tt] + sgi * kSeg, acc[tt]);
  }
  for (int tt = 0; tt < nt; ++tt) out[tt] = reduce_avx2(acc[tt]);
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

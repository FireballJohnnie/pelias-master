// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
#include "quant/tdecode.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

#include "quant/quant_cpu.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define PELIAS_TDECODE_X86 1
#endif

namespace pelias::quant {

void act_planes_32(const std::int8_t* q, std::uint32_t plane[8]) {
  for (int k = 0; k < 8; ++k) plane[k] = 0;
  for (int i = 0; i < 32; ++i) {
    const auto u = static_cast<std::uint32_t>(static_cast<int>(q[i]) + 128);
    for (int k = 0; k < 8; ++k) plane[k] |= ((u >> k) & 1u) << i;
  }
}

#ifdef PELIAS_TDECODE_X86
__attribute__((target("avx2"))) static void planes_avx2(const std::int8_t* q, std::uint32_t plane[8]) {
  // xu = x + 128 is x with its top bit flipped; shifting each 16-bit lane left by 7 - k puts bit k of both bytes at
  // their bit 7 (bit 15 for the high byte), and pmovmskb collects bit 7 of all 32 bytes: one plane per shift.
  __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
  v = _mm256_xor_si256(v, _mm256_set1_epi8(static_cast<char>(0x80)));
  plane[7] = static_cast<std::uint32_t>(_mm256_movemask_epi8(v));
  plane[6] = static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(v, 1)));
  plane[5] = static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(v, 2)));
  plane[4] = static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(v, 3)));
  plane[3] = static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(v, 4)));
  plane[2] = static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(v, 5)));
  plane[1] = static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(v, 6)));
  plane[0] = static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(v, 7)));
}
#endif

bool act_planes_32_fast(const std::int8_t* q, std::uint32_t plane[8]) {
#ifdef PELIAS_TDECODE_X86
  if (__builtin_cpu_supports("avx2")) {
    planes_avx2(q, plane);
    return true;
  }
#endif
  (void)q;
  (void)plane;
  return false;
}

void act_quantize_32(const float* x, std::int64_t n, ActBlock32* out) {
  for (std::int64_t b = 0; b < n / 32; ++b, x += 32) {
    ActBlock32& o = out[b];
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    o.scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / o.scale;
    o.sum = 0;
    for (int i = 0; i < 32; ++i) {
      o.q[i] = static_cast<std::int8_t>(std::min(127.0f, std::max(-127.0f, std::nearbyint(x[i] * inv))));
      o.sum += o.q[i];
    }
    if (!act_planes_32_fast(o.q, o.plane)) act_planes_32(o.q, o.plane);
  }
}

void q7_zt_split(const std::uint8_t* in, std::int64_t n, std::uint8_t* base, std::uint8_t* hbits) {
  for (std::int64_t i = 0; i < n / 256; ++i) {
    std::memcpy(base + kQ7ZTBaseBytes * i, in + 241 * i, kQ7ZTBaseBytes);
    std::memcpy(hbits + kQ7ZTHBytes * i, in + 241 * i + kQ7ZTBaseBytes, kQ7ZTHBytes);
  }
}

void q7_zt_join(const std::uint8_t* base, const std::uint8_t* hbits, std::int64_t n, std::uint8_t* out) {
  for (std::int64_t i = 0; i < n / 256; ++i) {
    std::memcpy(out + 241 * i, base + kQ7ZTBaseBytes * i, kQ7ZTBaseBytes);
    std::memcpy(out + 241 * i + kQ7ZTBaseBytes, hbits + kQ7ZTHBytes * i, kQ7ZTHBytes);
  }
}

float q7_zt_dot(const std::uint8_t* base, const std::uint8_t* hbits, const ActBlock32* x, std::int64_t n, HRoute route,
                std::int64_t* part2) {
  float acc = 0.0f;
  for (std::int64_t i = 0; i < n / 256; ++i, base += kQ7ZTBaseBytes) {
    const float d = fp16_to_fp32(static_cast<std::uint16_t>(base[0] | (base[1] << 8)));
    std::uint64_t lv = 0;
    for (int b = 0; b < 6; ++b) lv |= static_cast<std::uint64_t>(base[2 + b]) << (8 * b);
    const std::uint8_t mask = base[8];
    int hslot = 0;
    for (int j = 0; j < 8; ++j) {
      const ActBlock32& a = x[8 * i + j];
      const bool gated = (mask >> j) & 1u;
      std::uint32_t h = 0;
      if (gated && route != HRoute::missing) {
        const std::uint8_t* hp = hbits + kQ7ZTHBytes * i + 4 * hslot;
        h = static_cast<std::uint32_t>(hp[0]) | (static_cast<std::uint32_t>(hp[1]) << 8) | (static_cast<std::uint32_t>(hp[2]) << 16) |
            (static_cast<std::uint32_t>(hp[3]) << 24);
      }
      if (gated) ++hslot;
      std::int32_t B = 0, F = 0;  // base dot; fold dot
      for (int g = 0; g < 4; ++g) {
        std::uint64_t wv = 0;
        for (int b = 0; b < 7; ++b) wv |= static_cast<std::uint64_t>(base[9 + 7 * (4 * j + g) + b]) << (8 * b);
        for (int k = 0; k < 8; ++k) {
          const int idx = 8 * g + k;
          const int bb = static_cast<int>((wv >> (7 * k)) & 127u);
          B += (2 * bb - 127) * a.q[idx];
          if (route == HRoute::fold) F += (2 * bb + static_cast<int>((h >> idx) & 1u) - 127) * a.q[idx];
        }
      }
      std::int64_t P2;  // twice the integer partial sum
      if (route == HRoute::fold) {
        P2 = 2 * static_cast<std::int64_t>(F);
      } else if (!gated) {
        P2 = 2 * static_cast<std::int64_t>(B);
      } else if (route == HRoute::one_bit) {
        std::int64_t hc = -128 * std::popcount(h);
        for (int k = 0; k < 8; ++k) hc += static_cast<std::int64_t>(std::popcount(h & a.plane[k])) << k;
        P2 = 2 * (static_cast<std::int64_t>(B) + hc);
      } else {
        P2 = 2 * static_cast<std::int64_t>(B) + a.sum;  // H missing: + 0.5 * sum(x)
      }
      if (part2) part2[8 * i + j] = P2;
      const float s = q8_z_scale(d, static_cast<int>((lv >> (6 * j)) & 63u));
      acc += s * a.scale * (0.5f * static_cast<float>(P2));
    }
  }
  return acc;
}

}  // namespace pelias::quant
